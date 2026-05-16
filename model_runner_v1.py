if self.pcp_size > 1:
                self.pcp_manager.remap_mrope_positions_for_pcp(
                    positions_np,
                    num_scheduled_tokens,
                    num_reqs,
                    self.input_batch,
                    self.requests,
                    self.mrope_positions,
                )
