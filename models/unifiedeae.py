# paie model
import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel
from utils import hungarian_matcher, get_best_span, get_best_span_simple

SMALL = 1e-08
EPS = 1e-12

class UnifiedEAE(nn.Module):
    def __init__(self, config ,path):
        super(UnifiedEAE, self).__init__()
        self.config = config
        self.model_shared = BartModel.from_pretrained(path)
        self.model_a = BartModel.from_pretrained(path)
        self.model_b = BartModel.from_pretrained(path)
        self.w_prompt_start_shared = nn.Parameter(torch.rand(config.d_model, ))
        self.w_prompt_end_shared = nn.Parameter(torch.rand(config.d_model, ))
        self.w_prompt_start_shared_orig = nn.Parameter(torch.rand(config.d_model, ))
        self.w_prompt_end_shared_orig = nn.Parameter(torch.rand(config.d_model, ))
        self.model_shared._init_weights(self.w_prompt_start_shared)
        self.model_shared._init_weights(self.w_prompt_end_shared)
        self.model_shared._init_weights(self.w_prompt_start_shared_orig)
        self.model_shared._init_weights(self.w_prompt_end_shared_orig)

        self.w_prompt_start_dataset_a = nn.Parameter(torch.rand(config.d_model, ))
        self.w_prompt_end_dataset_a = nn.Parameter(torch.rand(config.d_model, ))
        self.model_a._init_weights(self.w_prompt_start_dataset_a)
        self.model_a._init_weights(self.w_prompt_end_dataset_a)

        self.w_prompt_start_dataset_b = nn.Parameter(torch.rand(config.d_model, ))
        self.w_prompt_end_dataset_b = nn.Parameter(torch.rand(config.d_model, ))
        self.model_b._init_weights(self.w_prompt_start_dataset_b)
        self.model_b._init_weights(self.w_prompt_end_dataset_b)

        self.loss_fct = nn.CrossEntropyLoss(reduction='sum')

        self.dropout_layer = nn.Dropout(0.1)
        self.tag_dim = 768
        self.hidden_dim = 768
        self.sample_size = 5
        self.max_seq_len = 512
        self.hidden2mean = nn.Linear(self.hidden_dim, self.tag_dim)
        self.hidden2std = nn.Linear(self.hidden_dim, self.tag_dim)
        self.beta = 0.000001
        self.alpha = 0.01
        self.alpha_init = 0.01
        self.r_var = self.tag_dim * self.max_seq_len
        self.r_mean = nn.Parameter(torch.randn(self.max_seq_len, self.tag_dim))
        self.r_std = nn.Parameter(torch.randn(self.max_seq_len, self.tag_dim))
        self.w_a = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.w_b = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.w_decoder_a = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.w_decoder_b = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.classifier = nn.Linear(self.tag_dim, 2)
        self.activation = nn.Sigmoid()
        self.criterion = nn.CrossEntropyLoss()

    def forward_sent_batch(self, hidden):
        mean = self.hidden2mean(hidden)  # bsz, seqlen, dim
        std = self.hidden2std(hidden)  # bsz, seqlen, dim
        cov = std * std + SMALL
        return mean, cov

    def get_sample_from_param_batch(self, mean, cov, sample_size):
        bsz, seqlen, tag_dim = mean.shape
        z = torch.randn(bsz, sample_size, seqlen, tag_dim).to("cuda")
        z = z * torch.sqrt(cov).unsqueeze(1).expand(-1, sample_size, -1, -1) + \
            mean.unsqueeze(1).expand(-1, sample_size, -1, -1)
        return z#.view(-1, seqlen, tag_dim)

    def get_statistics_batch(self, hidden):
        mean, cov = self.forward_sent_batch(hidden)
        return mean, cov

    def kl_div(self, param1, param2):
        mean1, cov1 = param1
        mean2, cov2 = param2
        bsz, seqlen, tag_dim = mean1.shape
        var_len = tag_dim * seqlen

        cov2_inv = 1 / cov2
        mean_diff = mean1 - mean2

        mean_diff = mean_diff.view(bsz, -1)
        cov1 = cov1.view(bsz, -1)
        cov2 = cov2.view(bsz, -1)
        cov2_inv = cov2_inv.view(bsz, -1)

        temp = (mean_diff * cov2_inv).view(bsz, 1, -1)
        KL = 0.5 * (torch.sum(torch.log(cov2), dim=1) - torch.sum(torch.log(cov1), dim=1) - var_len
                    + torch.sum(cov2_inv * cov1, dim=1) + torch.bmm(temp, mean_diff.view(bsz, -1, 1)).view(bsz))
        return KL

    def forward(
            self,
            enc_input_ids=None,
            enc_mask_ids=None,
            dec_prompt_ids=None,
            dec_prompt_mask_ids=None,
            arg_joint_prompts=None,
            target_info=None,
            old_tok_to_new_tok_indexs=None,
            arg_list=None,
            dataset_ids=None,
    ):
        """
        Args:
            multi args post calculation
        """
        if self.config.context_representation == 'decoder':
            context_outputs_shared = self.model_shared(
                enc_input_ids,
                attention_mask=enc_mask_ids,
                return_dict=True,
            )
            decoder_context_shared = context_outputs_shared.encoder_last_hidden_state
            context_outputs_shared = context_outputs_shared.last_hidden_state

            if torch.sum(dataset_ids == 0) > 0:
                context_outputs_dataset_a = self.model_a(
                    enc_input_ids[dataset_ids == 0],
                    attention_mask=enc_mask_ids[dataset_ids == 0],
                    return_dict=True,
                )
                decoder_context_dataset_a = context_outputs_dataset_a.encoder_last_hidden_state
                context_outputs_dataset_a = context_outputs_dataset_a.last_hidden_state

            if torch.sum(dataset_ids == 1) > 0:
                context_outputs_dataset_b = self.model_b(
                    enc_input_ids[dataset_ids == 1],
                    attention_mask=enc_mask_ids[dataset_ids == 1],
                    return_dict=True,
                )
                decoder_context_dataset_b = context_outputs_dataset_b.encoder_last_hidden_state
                context_outputs_dataset_b = context_outputs_dataset_b.last_hidden_state
        else:
            context_outputs_shared = self.model_shared.encoder(
                enc_input_ids,
                attention_mask=enc_mask_ids,
            )
            context_outputs_shared = context_outputs_shared.last_hidden_state
            decoder_context_shared = context_outputs_shared

            if torch.sum(dataset_ids == 0) > 0:
                context_outputs_dataset_a = self.model_a.encoder(
                    enc_input_ids[dataset_ids == 0],
                    attention_mask=enc_mask_ids[dataset_ids == 0],
                )
                context_outputs_dataset_a = context_outputs_dataset_a.last_hidden_state
                decoder_context_dataset_a = context_outputs_dataset_a
            if torch.sum(dataset_ids == 1) > 0:
                context_outputs_dataset_b = self.model_b.encoder(
                    enc_input_ids[dataset_ids == 1],
                    attention_mask=enc_mask_ids[dataset_ids == 1],
                )
                context_outputs_dataset_b = context_outputs_dataset_b.last_hidden_state
                decoder_context_dataset_b = context_outputs_dataset_b

        decoder_prompt_outputs_shared = self.model_shared.decoder(
            input_ids=dec_prompt_ids,
            attention_mask=dec_prompt_mask_ids,
            encoder_hidden_states=decoder_context_shared,
            encoder_attention_mask=enc_mask_ids,
        )
        decoder_prompt_outputs_shared = decoder_prompt_outputs_shared.last_hidden_state  # [bs, prompt_len, H]

        if torch.sum(dataset_ids == 0) > 0:
            decoder_prompt_outputs_dataset_a = self.model_a.decoder(
                input_ids=dec_prompt_ids[dataset_ids == 0],
                attention_mask=dec_prompt_mask_ids[dataset_ids == 0],
                encoder_hidden_states=decoder_context_dataset_a,
                encoder_attention_mask=enc_mask_ids[dataset_ids == 0],
            )
            decoder_prompt_outputs_dataset_a = decoder_prompt_outputs_dataset_a.last_hidden_state  # [bs, prompt_len, H]
        if torch.sum(dataset_ids == 1) > 0:
            decoder_prompt_outputs_dataset_b = self.model_b.decoder(
                input_ids=dec_prompt_ids[dataset_ids == 1],
                attention_mask=dec_prompt_mask_ids[dataset_ids == 1],
                encoder_hidden_states=decoder_context_dataset_b,
                encoder_attention_mask=enc_mask_ids[dataset_ids == 1],
            )
            decoder_prompt_outputs_dataset_b = decoder_prompt_outputs_dataset_b.last_hidden_state  # [bs, prompt_len, H]

        logit_lists_shared = list()
        logit_lists_dataset_a = list()
        logit_lists_dataset_b = list()
        total_loss_shared = 0.
        total_loss_shared_orig = 0.
        total_loss_dataset_a = 0
        total_loss_dataset_b = 0
        mean, cov = self.get_statistics_batch(context_outputs_shared)
        bsz, seqlen = context_outputs_shared.shape[0], context_outputs_shared.shape[1]

        if self.training:
            t = self.get_sample_from_param_batch(mean, cov, self.sample_size)
            # t = mean
        else:
            t = mean
        for i, (context_output, decoder_prompt_output, arg_joint_prompt, old_tok_to_new_tok_index) in \
                enumerate(zip(context_outputs_shared, decoder_prompt_outputs_shared, arg_joint_prompts,
                              old_tok_to_new_tok_indexs)):

            batch_loss = list()
            cnt = 0
            batch_loss_orig = list()
            cnt_orig = 0

            if not self.training:
                output = dict()
                for arg_role in arg_joint_prompt.keys():
                    """
                    "arg_role": {"tok_s": , "tok_e": }
                    """
                    prompt_slots = arg_joint_prompt[arg_role]
                    start_logits_list = list()
                    end_logits_list = list()
                    for (p_start, p_end) in zip(prompt_slots['tok_s'], prompt_slots['tok_e']):
                        prompt_query_sub = decoder_prompt_output[p_start:p_end]
                        prompt_query_sub = torch.mean(prompt_query_sub, dim=0).unsqueeze(0)

                        start_query = (prompt_query_sub * self.w_prompt_start_shared).unsqueeze(-1)  # [1, H, 1]
                        end_query = (prompt_query_sub * self.w_prompt_end_shared).unsqueeze(-1)  # [1, H, 1]

                        start_logits = torch.bmm(t[i].unsqueeze(0), start_query).squeeze()
                        end_logits = torch.bmm(t[i].unsqueeze(0), end_query).squeeze()

                        start_logits_list.append(start_logits)
                        end_logits_list.append(end_logits)

                    output[arg_role] = [start_logits_list, end_logits_list]
                logit_lists_shared.append(output)
            else:
                for arg_role in arg_joint_prompt.keys():
                    """
                    "arg_role": {"tok_s": , "tok_e": }
                    """
                    prompt_slots = arg_joint_prompt[arg_role]
                    start_logits_list = list()
                    end_logits_list = list()
                    for (p_start, p_end) in zip(prompt_slots['tok_s'], prompt_slots['tok_e']):
                        prompt_query_sub = decoder_prompt_output[p_start:p_end]
                        prompt_query_sub = torch.mean(prompt_query_sub, dim=0).unsqueeze(0)

                        start_query = (prompt_query_sub * self.w_prompt_start_shared_orig).unsqueeze(-1)  # [1, H, 1]
                        end_query = (prompt_query_sub * self.w_prompt_end_shared_orig).unsqueeze(-1)  # [1, H, 1]

                        start_logits = torch.bmm(context_output.unsqueeze(0), start_query).squeeze()
                        end_logits = torch.bmm(context_output.unsqueeze(0), end_query).squeeze()

                        start_logits_list.append(start_logits)
                        end_logits_list.append(end_logits)
                    target = target_info[i][arg_role]  # "arg_role": {"text": ,"span_s": ,"span_e": }
                    predicted_spans = list()
                    for (start_logits, end_logits) in zip(start_logits_list, end_logits_list):
                        if self.config.matching_method_train == 'accurate':
                            predicted_spans.append(get_best_span(start_logits, end_logits, old_tok_to_new_tok_index,
                                                                 self.config.max_span_length))
                        elif self.config.matching_method_train == 'max':
                            predicted_spans.append(get_best_span_simple(start_logits, end_logits))
                        else:
                            raise AssertionError()

                    target_spans = [[s, e] for (s, e) in zip(target["span_s"], target["span_e"])]
                    if len(target_spans) < len(predicted_spans):
                        # need to consider whether to make more
                        pad_len = len(predicted_spans) - len(target_spans)
                        target_spans = target_spans + [[0, 0]] * pad_len
                        target["span_s"] = target["span_s"] + [0] * pad_len
                        target["span_e"] = target["span_e"] + [0] * pad_len

                    if self.config.bipartite:
                        idx_preds, idx_targets = hungarian_matcher(predicted_spans, target_spans)
                    else:
                        idx_preds = list(range(len(predicted_spans)))
                        idx_targets = list(range(len(target_spans)))
                        if len(idx_targets) > len(idx_preds):
                            idx_targets = idx_targets[0:len(idx_preds)]
                        idx_preds = torch.as_tensor(idx_preds, dtype=torch.int64)
                        idx_targets = torch.as_tensor(idx_targets, dtype=torch.int64)

                    cnt_orig += len(idx_preds)
                    start_loss = self.loss_fct(torch.stack(start_logits_list)[idx_preds],
                                               torch.LongTensor(target["span_s"]).to(self.config.device)[idx_targets])
                    end_loss = self.loss_fct(torch.stack(end_logits_list)[idx_preds],
                                             torch.LongTensor(target["span_e"]).to(self.config.device)[idx_targets])
                    batch_loss_orig.append((start_loss + end_loss) / 2)

                for sample_num in range(self.sample_size):
                    for arg_role in arg_joint_prompt.keys():
                        """
                        "arg_role": {"tok_s": , "tok_e": }
                        """
                        prompt_slots = arg_joint_prompt[arg_role]
                        start_logits_list = list()
                        end_logits_list = list()
                        for (p_start, p_end) in zip(prompt_slots['tok_s'], prompt_slots['tok_e']):
                            prompt_query_sub = decoder_prompt_output[p_start:p_end]
                            prompt_query_sub = torch.mean(prompt_query_sub, dim=0).unsqueeze(0)
                            # print(prompt_query_sub.size(), self.w_prompt_start_shared.size())
                            start_query = (prompt_query_sub * self.w_prompt_start_shared).unsqueeze(-1)  # [1, H, 1]
                            end_query = (prompt_query_sub * self.w_prompt_end_shared).unsqueeze(-1)  # [1, H, 1]
                            # print("sample_num: ", sample_num, "t:" , t.size(), t[i, sample_num].unsqueeze(0).size(), start_query.size())
                            start_logits = torch.bmm(t[i, sample_num].unsqueeze(0), start_query).squeeze()
                            end_logits = torch.bmm(t[i, sample_num].unsqueeze(0), end_query).squeeze()

                            start_logits_list.append(start_logits)
                            end_logits_list.append(end_logits)

                        target = target_info[i][arg_role]  # "arg_role": {"text": ,"span_s": ,"span_e": }
                        predicted_spans = list()
                        for (start_logits, end_logits) in zip(start_logits_list, end_logits_list):
                            if self.config.matching_method_train == 'accurate':
                                predicted_spans.append(get_best_span(start_logits, end_logits, old_tok_to_new_tok_index,
                                                                     self.config.max_span_length))
                            elif self.config.matching_method_train == 'max':
                                predicted_spans.append(get_best_span_simple(start_logits, end_logits))
                            else:
                                raise AssertionError()

                        target_spans = [[s, e] for (s, e) in zip(target["span_s"], target["span_e"])]
                        if len(target_spans) < len(predicted_spans):
                            # need to consider whether to make more
                            pad_len = len(predicted_spans) - len(target_spans)
                            target_spans = target_spans + [[0, 0]] * pad_len
                            target["span_s"] = target["span_s"] + [0] * pad_len
                            target["span_e"] = target["span_e"] + [0] * pad_len

                        if self.config.bipartite:
                            idx_preds, idx_targets = hungarian_matcher(predicted_spans, target_spans)
                        else:
                            idx_preds = list(range(len(predicted_spans)))
                            idx_targets = list(range(len(target_spans)))
                            if len(idx_targets) > len(idx_preds):
                                idx_targets = idx_targets[0:len(idx_preds)]
                            idx_preds = torch.as_tensor(idx_preds, dtype=torch.int64)
                            idx_targets = torch.as_tensor(idx_targets, dtype=torch.int64)

                        cnt += len(idx_preds)
                        start_loss = self.loss_fct(torch.stack(start_logits_list)[idx_preds],
                                                   torch.LongTensor(target["span_s"]).to(self.config.device)[idx_targets])
                        end_loss = self.loss_fct(torch.stack(end_logits_list)[idx_preds],
                                                 torch.LongTensor(target["span_e"]).to(self.config.device)[idx_targets])
                        batch_loss.append((start_loss + end_loss) / 2)

            if self.training:  # inside batch mean loss
                total_loss_shared_orig = total_loss_shared_orig + torch.sum(torch.stack(batch_loss_orig)) / cnt_orig
                total_loss_shared = total_loss_shared + torch.sum(torch.stack(batch_loss)) / cnt

        if self.training:
            mean_r = self.r_mean[:seqlen].unsqueeze(0).expand(bsz, -1, -1)
            std_r = self.r_std[:seqlen].unsqueeze(0).expand(bsz, -1, -1)
            cov_r = std_r * std_r + SMALL
            kl_div = self.kl_div((mean, cov), (mean_r, cov_r))
            loss_2 = self.beta * kl_div.mean()


        arg_joint_prompts_a = []
        arg_joint_prompts_b = []
        old_tok_to_new_tok_indexs_a = []
        old_tok_to_new_tok_indexs_b = []
        target_info_a = []
        target_info_b = []
        for i in range(dataset_ids.size(0)):
            if dataset_ids[i].item() == 0:
                arg_joint_prompts_a.append(arg_joint_prompts[i])
                old_tok_to_new_tok_indexs_a.append(old_tok_to_new_tok_indexs[i])
                if self.training:
                    target_info_a.append(target_info[i])
            if dataset_ids[i].item() == 1:
                arg_joint_prompts_b.append(arg_joint_prompts[i])
                old_tok_to_new_tok_indexs_b.append(old_tok_to_new_tok_indexs[i])
                if self.training:
                    target_info_b.append(target_info[i])
        # ----------------dataset_a----------------
        if torch.sum(dataset_ids == 0) > 0:
            weight_a = torch.sigmoid(self.w_a(torch.cat((context_outputs_dataset_a, context_outputs_shared[dataset_ids == 0]), dim=-1)))
            context_outputs_dataset_a = weight_a * context_outputs_dataset_a + (1-weight_a) * context_outputs_shared[dataset_ids == 0]
            weight_decoder_a = torch.sigmoid(self.w_decoder_a(torch.cat((decoder_prompt_outputs_dataset_a, decoder_prompt_outputs_shared[dataset_ids == 0]), dim=-1)))

            decoder_prompt_outputs_dataset_a = weight_decoder_a * decoder_prompt_outputs_dataset_a + (1-weight_decoder_a) * decoder_prompt_outputs_shared[
                dataset_ids == 0]
            for i, (context_output, decoder_prompt_output, arg_joint_prompt, old_tok_to_new_tok_index) in \
                    enumerate(zip(context_outputs_dataset_a, decoder_prompt_outputs_dataset_a, arg_joint_prompts_a,
                                  old_tok_to_new_tok_indexs_a)):

                batch_loss = list()
                cnt = 0

                output = dict()
                for arg_role in arg_joint_prompt.keys():
                    """
                    "arg_role": {"tok_s": , "tok_e": }
                    """
                    prompt_slots = arg_joint_prompt[arg_role]

                    start_logits_list = list()
                    end_logits_list = list()
                    for (p_start, p_end) in zip(prompt_slots['tok_s'], prompt_slots['tok_e']):
                        prompt_query_sub = decoder_prompt_output[p_start:p_end]
                        prompt_query_sub = torch.mean(prompt_query_sub, dim=0).unsqueeze(0)

                        start_query = (prompt_query_sub * self.w_prompt_start_dataset_a).unsqueeze(-1)  # [1, H, 1]
                        end_query = (prompt_query_sub * self.w_prompt_end_dataset_a).unsqueeze(-1)  # [1, H, 1]

                        start_logits = torch.bmm(context_output.unsqueeze(0), start_query).squeeze()
                        end_logits = torch.bmm(context_output.unsqueeze(0), end_query).squeeze()

                        start_logits_list.append(start_logits)
                        end_logits_list.append(end_logits)

                    output[arg_role] = [start_logits_list, end_logits_list]

                    if self.training:
                        # calculate loss
                        target = target_info_a[i][arg_role]  # "arg_role": {"text": ,"span_s": ,"span_e": }
                        predicted_spans = list()
                        for (start_logits, end_logits) in zip(start_logits_list, end_logits_list):
                            if self.config.matching_method_train == 'accurate':
                                predicted_spans.append(get_best_span(start_logits, end_logits, old_tok_to_new_tok_index,
                                                                     self.config.max_span_length))
                            elif self.config.matching_method_train == 'max':
                                predicted_spans.append(get_best_span_simple(start_logits, end_logits))
                            else:
                                raise AssertionError()

                        target_spans = [[s, e] for (s, e) in zip(target["span_s"], target["span_e"])]
                        if len(target_spans) < len(predicted_spans):
                            # need to consider whether to make more
                            pad_len = len(predicted_spans) - len(target_spans)
                            target_spans = target_spans + [[0, 0]] * pad_len
                            target["span_s"] = target["span_s"] + [0] * pad_len
                            target["span_e"] = target["span_e"] + [0] * pad_len

                        if self.config.bipartite:
                            idx_preds, idx_targets = hungarian_matcher(predicted_spans, target_spans)
                        else:
                            idx_preds = list(range(len(predicted_spans)))
                            idx_targets = list(range(len(target_spans)))
                            if len(idx_targets) > len(idx_preds):
                                idx_targets = idx_targets[0:len(idx_preds)]
                            idx_preds = torch.as_tensor(idx_preds, dtype=torch.int64)
                            idx_targets = torch.as_tensor(idx_targets, dtype=torch.int64)

                        cnt += len(idx_preds)
                        start_loss = self.loss_fct(torch.stack(start_logits_list)[idx_preds],
                                                   torch.LongTensor(target["span_s"]).to(self.config.device)[
                                                       idx_targets])
                        end_loss = self.loss_fct(torch.stack(end_logits_list)[idx_preds],
                                                 torch.LongTensor(target["span_e"]).to(self.config.device)[idx_targets])
                        batch_loss.append((start_loss + end_loss) / 2)

                logit_lists_dataset_a.append(output)
                if self.training:  # inside batch mean loss
                    total_loss_dataset_a = total_loss_dataset_a + torch.sum(torch.stack(batch_loss)) / cnt

        # ----------------dataset_b----------------
        if torch.sum(dataset_ids == 1) > 0:
            # print(dataset_ids)
            # print(torch.sum(dataset_ids == 1))
            # print(decoder_prompt_outputs_dataset_b.size())
            # print(context_outputs_dataset_b.size())
            weight_b = torch.sigmoid(
                self.w_b(torch.cat((context_outputs_dataset_b, context_outputs_shared[dataset_ids == 1]), dim=-1)))
            context_outputs_dataset_b = weight_b * context_outputs_dataset_b + (1 - weight_b) * context_outputs_shared[
                dataset_ids == 1]
            weight_decoder_b = torch.sigmoid(self.w_decoder_b(
                torch.cat((decoder_prompt_outputs_dataset_b, decoder_prompt_outputs_shared[dataset_ids == 1]), dim=-1)))

            decoder_prompt_outputs_dataset_b = weight_decoder_b * decoder_prompt_outputs_dataset_b + (
                        1 - weight_decoder_b) * decoder_prompt_outputs_shared[
                                                   dataset_ids == 1]

            # context_outputs_dataset_b = weight_dataset_b * context_outputs_dataset_b + (1-weight_dataset_b) * context_outputs_shared[dataset_ids == 1]
            # decoder_prompt_outputs_dataset_b = weight_dataset_b * decoder_prompt_outputs_dataset_b + (1-weight_dataset_b) * decoder_prompt_outputs_shared[
            #     dataset_ids == 1]
            for i, (context_output, decoder_prompt_output, arg_joint_prompt, old_tok_to_new_tok_index) in \
                    enumerate(zip(context_outputs_dataset_b, decoder_prompt_outputs_dataset_b, arg_joint_prompts_b,
                                  old_tok_to_new_tok_indexs_b)):

                batch_loss = list()
                cnt = 0

                output = dict()
                for arg_role in arg_joint_prompt.keys():
                    """
                    "arg_role": {"tok_s": , "tok_e": }
                    """
                    prompt_slots = arg_joint_prompt[arg_role]

                    start_logits_list = list()
                    end_logits_list = list()
                    for (p_start, p_end) in zip(prompt_slots['tok_s'], prompt_slots['tok_e']):
                        prompt_query_sub = decoder_prompt_output[p_start:p_end]
                        prompt_query_sub = torch.mean(prompt_query_sub, dim=0).unsqueeze(0)

                        start_query = (prompt_query_sub * self.w_prompt_start_dataset_b).unsqueeze(-1)  # [1, H, 1]
                        end_query = (prompt_query_sub * self.w_prompt_end_dataset_b).unsqueeze(-1)  # [1, H, 1]

                        start_logits = torch.bmm(context_output.unsqueeze(0), start_query).squeeze()
                        end_logits = torch.bmm(context_output.unsqueeze(0), end_query).squeeze()

                        start_logits_list.append(start_logits)
                        end_logits_list.append(end_logits)

                    output[arg_role] = [start_logits_list, end_logits_list]

                    if self.training:
                        # calculate loss
                        target = target_info_b[i][arg_role]  # "arg_role": {"text": ,"span_s": ,"span_e": }
                        predicted_spans = list()
                        for (start_logits, end_logits) in zip(start_logits_list, end_logits_list):
                            if self.config.matching_method_train == 'accurate':
                                predicted_spans.append(
                                    get_best_span(start_logits, end_logits, old_tok_to_new_tok_index,
                                                  self.config.max_span_length))
                            elif self.config.matching_method_train == 'max':
                                predicted_spans.append(get_best_span_simple(start_logits, end_logits))
                            else:
                                raise AssertionError()

                        target_spans = [[s, e] for (s, e) in zip(target["span_s"], target["span_e"])]
                        if len(target_spans) < len(predicted_spans):
                            # need to consider whether to make more
                            pad_len = len(predicted_spans) - len(target_spans)
                            target_spans = target_spans + [[0, 0]] * pad_len
                            target["span_s"] = target["span_s"] + [0] * pad_len
                            target["span_e"] = target["span_e"] + [0] * pad_len

                        if self.config.bipartite:
                            idx_preds, idx_targets = hungarian_matcher(predicted_spans, target_spans)
                        else:
                            idx_preds = list(range(len(predicted_spans)))
                            idx_targets = list(range(len(target_spans)))
                            if len(idx_targets) > len(idx_preds):
                                idx_targets = idx_targets[0:len(idx_preds)]
                            idx_preds = torch.as_tensor(idx_preds, dtype=torch.int64)
                            idx_targets = torch.as_tensor(idx_targets, dtype=torch.int64)

                        cnt += len(idx_preds)
                        start_loss = self.loss_fct(torch.stack(start_logits_list)[idx_preds],
                                                   torch.LongTensor(target["span_s"]).to(self.config.device)[
                                                       idx_targets])
                        end_loss = self.loss_fct(torch.stack(end_logits_list)[idx_preds],
                                                 torch.LongTensor(target["span_e"]).to(self.config.device)[
                                                     idx_targets])
                        batch_loss.append((start_loss + end_loss) / 2)

                logit_lists_dataset_b.append(output)
                if self.training:  # inside batch mean loss
                    total_loss_dataset_b = total_loss_dataset_b + torch.sum(torch.stack(batch_loss)) / cnt
        logit_lists_dataset = []
        dataset_a_index = 0
        dataset_b_index = 0
        for i in range(dataset_ids.size(0)):
            if dataset_ids[i].item() == 0:
                logit_lists_dataset.append(logit_lists_dataset_a[dataset_a_index])
                dataset_a_index += 1
            elif dataset_ids[i].item() == 1:
                logit_lists_dataset.append(logit_lists_dataset_b[dataset_b_index])
                dataset_b_index += 1
        if self.training:
            total_loss_shared = total_loss_shared / len(context_outputs_shared) + loss_2
            total_loss_shared_orig = total_loss_shared_orig / len(context_outputs_shared)
            dataset_num = 0
            if torch.sum(dataset_ids == 0) > 0:
                dataset_num += 1
                total_loss_dataset_a = total_loss_dataset_a / len(context_outputs_dataset_a)
            else:
                total_loss_dataset_a = 0
            if torch.sum(dataset_ids == 1) > 0:
                dataset_num += 1
                total_loss_dataset_b = total_loss_dataset_b / len(context_outputs_dataset_b)
            else:
                total_loss_dataset_b = 0
            total_loss = self.alpha*total_loss_shared + total_loss_shared_orig + total_loss_dataset_a + total_loss_dataset_b
            # print("&&"*10, total_loss, total_loss_shared, total_loss_dataset_a, total_loss_dataset_b, dataset_a_index, dataset_b_index, dataset_ids)
            # print(dataset_ids)
            return total_loss, logit_lists_shared, logit_lists_dataset
        else:
            return [], logit_lists_shared, logit_lists_dataset
