logits_indices = (
                torch.from_numpy(cu_num_tokens) * self.pcp_world_size
                - self.num_pcp_pads_cpu_tensor[: self.num_reqs]
                - 1
            )
            cu_num_tokens_tensor = torch.from_numpy(cu_num_tokens)
            logits_indices = cu_num_tokens_tensor * self.pcp_world_size - self.num_pcp_pads_cpu_tensor[: self.num_reqs] - 1
            if self.num_dycp_reqs == 0:
                logits_indices[self.num_dycp_reqs: self.num_reqs] = cu_num_tokens_tensor - 1
            else:
                logits_indices[self.num_dycp_reqs: self.num_reqs] = cu_num_tokens_tensor[
                    self.num_dycp_reqs - 1] * self.pcp_world_size + (cu_num_tokens_tensor[self.num_dycp_reqs:] - cu_num_tokens_tensor[self.num_dycp_reqs - 1]) - 1
