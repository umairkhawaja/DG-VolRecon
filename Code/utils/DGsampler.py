import numpy as np
import torch
from torch.special import erf
from einops import (rearrange, reduce, repeat)
from .grid_sample import grid_sample_2d

class DGSampler():
    """
    NeRF-like sampler that sample denser around the surface
    """
    def __init__(self, point_num=64, n_candidate=512, alpha=-1, n_gaussian=20):
        self.point_num = point_num
        self.n_candidate = n_candidate
        self.alpha = alpha
        self.n_gaussian = n_gaussian

    def sample_coarse(self, ray_o, ray_d, jitter=True, near_z=None, far_z=None):
        """
        return: [Bs, n_ray, n_candidate]
        """
        device = ray_o.device
        step = 1.0 / self.n_candidate
        B, RN, _ = ray_o.shape
        near_z = near_z.reshape(-1, 1)
        far_z = far_z.reshape(-1, 1)
        z_steps = torch.linspace(0, 1 - step, self.n_candidate, device=device)  # (Kc)
        z_steps = z_steps.unsqueeze(0).repeat(B*RN, 1)  # (B, Kc)
        z_steps += torch.rand_like(z_steps) * step

        # Use linear sampling in depth space
        z_samples = near_z * (1 - z_steps) + far_z * z_steps  # (B, Kc)
        z_samples = z_samples.reshape(B, RN, self.n_candidate) # B, RN, n_candidates
        return z_samples

    def fill_up_uniform_samples(self, z_samples, ray_o, ray_d, near_z=None, far_z=None):
        """
        Fills up empty slots in samples (indicated by 0) uniformly
        :param z_samples: z values of existing samples (B, n_rays, n_samples). Empty samples have value 0
        :return: filled up z_samples
        """

        # preparing data to calculate remaining z_samples in parallel
        B, RN, _ = ray_o.shape
        z_samples = z_samples.sort(dim=-1).values  # zeros in the front, important for parallelized filling
        z_samples = z_samples.view(-1, z_samples.shape[-1])  # (B RN), n_samples
        near_z = near_z.reshape(-1, 1)
        far_z = far_z.reshape(-1, 1)
        sample_missing_mask = z_samples == 0  # N_, n_coarsedepth
        missing_iray, missing_isample = torch.where(sample_missing_mask)  # (N_missing,)
        n_missing = sample_missing_mask.int().sum(dim=-1)  # (N_)
        n_missing = n_missing[missing_iray]  # (N_missing,)
        nears = near_z[missing_iray].squeeze(-1)  # (N_missing,)
        fars = far_z[missing_iray].squeeze(-1)  # (N_missing,)

        # calculating remaining z_samples
        step = (fars - nears) / n_missing  # (N_missing,)
        z_missing = nears + missing_isample * step  # (N_missing,)
        # z_missing += torch.rand_like(z_missing) * step

        # filling up
        z_samples[missing_iray, missing_isample] = z_missing
        z_samples = z_samples.reshape(B, RN, z_samples.shape[-1])

        z_samples = z_samples.sort(dim=-1).values  # sorted z sample values are required in self.composite()
        return z_samples

    def weighted_mean_n_std(self, x, weights, dim, keepdims=False):
        weights_normed = weights / weights.sum(dim=dim, keepdims=True)
        mean = (x * weights_normed).sum(dim=dim, keepdims=True)
        std = ((x - mean).pow(2) * weights_normed).sum(dim=dim, keepdims=True).sqrt()

        if not keepdims:
            mean = mean.squeeze(dim)
            std = std.squeeze(dim)
        return mean, std

    def dgsample(self, batch, ray_o, ray_d, jitter=True, near_z=None, far_z=None):
        """
        ray_o, ray_d: [(B RN) DimX]
        near_z, far_z: [(B RN)]
        """
        B, NV, _, _ = batch['source_poses'].shape
        depth_prior = batch["depths_prior_h"]
        depth_prob = batch['depths_prob']
        mid_z_val = - reduce(ray_o * ray_d, "RN Dim_X -> RN", 'sum')
        mid_z_val = rearrange(mid_z_val, "RN -> 1 RN")

        if near_z is None:
            # no near and far provided
            near = mid_z_val - 1.3#self.sample_radius
            far = mid_z_val + 1.3#self.sample_radius
        else:
            near = near_z
            far = far_z
        ray_d = rearrange(ray_d, "(B RN) DimX -> B RN DimX", B=B)
        ray_o = rearrange(ray_o, "(B RN) DimX -> B RN DimX", B=B)
        near = rearrange(near, "(aB RN) -> B RN", B=B)
        far = rearrange(far, "(B RN) -> B RN", B=B)
        RN = far.shape[1]
        z_cand = self.sample_coarse(ray_o, ray_d, near_z=near, far_z=far) # B, RN, n_candidates
        step_size = (far - near) / self.n_candidate  # B, NR
        ########### project 3d sampling points into pixel coord ##########
        xyz = ray_o[:, :, None] + z_cand.unsqueeze(-1) * ray_d[:, :, None] # B, RN, n_cand, 3
        xyz = repeat(xyz, "B RN SN DimX -> B NV RN SN DimX", NV=NV).float()
        xyz = xyz.reshape(B, NV, -1, 3)
        xyz = torch.cat([xyz, torch.ones_like(xyz[:, :, :, :1])], axis=3) # B, NV, RN*n_cand, 4
        xyz = rearrange(xyz, "B NV P DimX -> B NV DimX P")
        xyz = batch['source_poses'] @ xyz # B, NV, 4, RN*n_cand
        xyz = xyz[:, :, :3]
        uv = xyz[:, :, :2] / xyz[:, :, 2:3] # B, NV, 3, RN*n_cand
        uv = rearrange(uv, "B NV Dim2 XYZ -> B NV XYZ Dim2")
        projected_depth = xyz[:, :, 2:3] # B NV 1 RN*n_cand
        valid_depth_mask = projected_depth > 0
        ########### get pixel depth and valid mask ##########
        pixel_depth, pixel_depth_mask = grid_sample_2d(
            rearrange(depth_prior.unsqueeze(2), "B NV C H W -> (B NV) C H W"),
            rearrange(uv, "B NV (RN cand) Dim2-> (B NV) RN cand Dim2", RN=RN, cand=self.n_candidate).double())
        pixel_depth = rearrange(pixel_depth, '(B NV) C RN cand -> B NV C (RN cand)', B=B, NV=NV)
        pixel_depth_mask = rearrange(pixel_depth_mask.unsqueeze(1), '(B NV) C RN cand -> B NV C (RN cand)', B=B, NV=NV)
        ########### get pixel probability and valid mask ##########
        pixel_prob, pixel_prob_mask = grid_sample_2d(
            rearrange(depth_prob.unsqueeze(2), "B NV C H W -> (B NV) C H W"),
            rearrange(uv, "B NV (RN cand) Dim2-> (B NV) RN cand Dim2", RN=RN, cand=self.n_candidate))
        pixel_prob = rearrange(pixel_prob, '(B NV) C RN cand -> B NV C (RN cand)', B=B, NV=NV)
        pixel_prob_mask = rearrange(pixel_prob_mask.unsqueeze(1), '(B NV) C RN cand -> B NV C (RN cand)', B=B, NV=NV)
        pixel_std = torch.exp(self.alpha * pixel_prob)
        ########### calculate probability for each sampling point and shorlist ##########
        step_size = step_size.repeat_interleave(self.n_candidate, dim=1).view(B, 1, 1, RN * self.n_candidate)
        step_size = step_size.expand_as(pixel_depth)  # B, NV, 1, RN*n_cand
        pt_likelihood = torch.zeros_like(pixel_depth)  # B, NV, 1, RN*n_cand
        # diff = (pixel_depth - projected_depth).abs()

        depth_dist_mask = (pixel_depth - projected_depth).abs() < 1.5
        bg_mask = pixel_std != 1
        mask = (bg_mask.bool() &
                depth_dist_mask.bool() &
                valid_depth_mask.bool() &
                pixel_depth_mask.bool() &
                pixel_prob_mask.bool())  # B, NV, 1, RN*n_cand
        # num_true = torch.sum(mask).item()
        # print(num_true / mask.numel())
        pt_likelihood[mask] = 0.5 * (
                erf((projected_depth[mask] + step_size[mask] / 2 - pixel_depth[mask]) / (pixel_std[mask] * np.sqrt(2))) -
                erf((projected_depth[mask] - step_size[mask] / 2 - pixel_depth[mask]) / (pixel_std[mask] * np.sqrt(2)))
        ).abs()
        # pt_likelihood[mask] = 0.5 * (
        #         erf((projected_depth[mask] + step_size[mask] / 2 -  pixel_depth[mask]) / (
        #                     0.0002 * np.sqrt(2))) -
        #         erf((projected_depth[mask] - step_size[mask] / 2 - pixel_depth[mask]) / (0.0002 * np.sqrt(2)))
        # ).abs()
        pt_likelihood = torch.max(pt_likelihood, dim=1).values.squeeze(1)  # B, RN*n_cand
        pt_likelihood = pt_likelihood.reshape(B, RN, -1)  # B, N_rays, N_pointsperray
        opaque_pt_likelihood = pt_likelihood.clone()
        opaque_pt_likelihood[..., 1:] *= torch.cumprod(1. - pt_likelihood, dim=-1)[..., :-1]
        selected_pts_idcs = pt_likelihood.argsort(dim=-1, descending=True)[..., :self.point_num]  # B, N_rays, n_depthsmpls
        SB_helper = torch.arange(B).view(-1, 1, 1).expand_as(selected_pts_idcs)
        ray_helper = torch.arange(RN).view(1, -1, 1).expand_as(selected_pts_idcs)
        selected_pts_likelihood = pt_likelihood[SB_helper, ray_helper, selected_pts_idcs]  # SB, N_rays, n_depthsmpls
        zero_liklhd_mask = selected_pts_likelihood == 0.  # pts with 0 likelihood: z_sample=0 for filling up later
        z_samples_depth = z_cand[SB_helper, ray_helper, selected_pts_idcs]  # SB, N_rays, N_depthsamples
        z_samples_depth[zero_liklhd_mask] = 0  # no samples where no depth
        if self.n_gaussian > 0:
            ray_mask = torch.any(opaque_pt_likelihood != 0, dim=-1)  # SB, NR
            ray_dmean, ray_dstd = self.weighted_mean_n_std(z_cand[ray_mask],  # B, 1
                                                      opaque_pt_likelihood[ray_mask],
                                                      dim=-1, keepdims=True)
            gauss_samples = torch.zeros(*z_cand.shape[:-1], self.n_gaussian, device=z_cand.device,
                                        dtype=z_cand.dtype)
            gauss_samples[ray_mask] = torch.randn_like(gauss_samples[ray_mask]) * ray_dstd.float() + ray_dmean.float()
            # gauss samples: SB, NR, n_gaussian
            z_samples_depth[..., -self.n_gaussian:] = gauss_samples
        z_samples_depth = self.fill_up_uniform_samples(z_samples_depth, ray_o, ray_d, near, far)
        z_samples_depth = rearrange(z_samples_depth, 'B RN SN -> (B RN) SN')
        points_x = rearrange(ray_o, "B RN DimX -> (B RN) 1 DimX") + rearrange(z_samples_depth, "RN SN -> RN SN 1") * rearrange(ray_d,
                                                                                                               "B RN DimX -> (B RN) 1 DimX")
        points_d = repeat(ray_d, "B RN DimX -> (B RN) SN DimX", SN=self.point_num)
        sample_sort_idx = torch.sort(z_samples_depth, axis=1)[1]
        z_val_sorted = torch.gather(z_samples_depth, 1, sample_sort_idx)
        points_x_sorted = torch.gather(points_x, 1, repeat(sample_sort_idx, "RN SN -> RN SN 3"))

        return points_x_sorted, z_val_sorted.clone(), points_d


