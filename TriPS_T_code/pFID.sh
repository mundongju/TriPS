# python compute_patch_KID_final.py \
#   --label_dir /home/dongju1126/ICML_Triadic_Posterior/FlowDPS_Ours/workdir_phase_retrieval_demo_REBUTTAL_700_dc_100set_sto_flowdps/label \
#   --recon_dir /home/dongju1126/ICML_Triadic_Posterior/FlowDPS_Ours/workdir_phase_retrieval_demo_REBUTTAL_700_dc_100set_sto_flowdps/recon \
#   --patch_size 192 \
#   --kid_subset_size 1000 \
#   --kid_subsets 50

####### 100 set SR ######################
# python compute_patch_KID_final.py \
#   --label_dir /home/dongju1126/ICML_Triadic_Posterior/FlowDPS_Ours/workdir_Ours_reference_other_task_20260115/SR_FFHQ_100set_lin_log_exp_eta_0_dc_250_50/label \
#   --recon_dir /home/dongju1126/ICML_Triadic_Posterior/FlowDPS_Ours/workdir_Ours_reference_other_task_20260115/SR_FFHQ_100set_lin_log_exp_eta_0_dc_250_50/recon \
#   --patch_size 192 \
#   --kid_subset_size 1000 \
#   --kid_subsets 50

# python compute_patch_KID_final.py \
#   --label_dir /home/dongju1126/ICML_Triadic_Posterior/FlowDPS_Ours/workdir_Ours_reference_other_task_20260115/SR_DIV2K_100set_lin_log_exp_eta_0_dc_300_100/label \
#   --recon_dir /home/dongju1126/ICML_Triadic_Posterior/FlowDPS_Ours/workdir_Ours_reference_other_task_20260115/SR_DIV2K_100set_lin_log_exp_eta_0_dc_300_100/recon \
#   --patch_size 192 \
#   --kid_subset_size 1000 \
#   --kid_subsets 50

####### prompt ######################
# python compute_patch_KID_final.py \
#   --label_dir /home/dongju1126/ICML_Triadic_Posterior/FlowDPS_Ours/workdir_prompt_REBUTTAL/SR_DIV2K_null_prompt/label \
#   --recon_dir /home/dongju1126/ICML_Triadic_Posterior/FlowDPS_Ours/workdir_prompt_REBUTTAL/SR_DIV2K_null_prompt/recon \
#   --patch_size 192 \
#   --kid_subset_size 1000 \
#   --kid_subsets 50

# python compute_patch_KID_final.py \
#   --label_dir /home/dongju1126/ICML_Triadic_Posterior/FlowDPS_Ours/workdir_prompt_REBUTTAL/SR_FFHQ_misleading/label \
#   --recon_dir /home/dongju1126/ICML_Triadic_Posterior/FlowDPS_Ours/workdir_prompt_REBUTTAL/SR_FFHQ_misleading/recon \
#   --patch_size 192 \
#   --kid_subset_size 1000 \
#   --kid_subsets 50

####### phase retrieval ######################
# python compute_patch_KID_final.py \
#   --label_dir /home/dongju1126/ICML_Triadic_Posterior/FlowDPS_Ours/workdir_phase_retrieval_demo_REBUTTAL_700_dc_100set_top-k_k8/label \
#   --recon_dir /home/dongju1126/ICML_Triadic_Posterior/FlowDPS_Ours/workdir_phase_retrieval_demo_REBUTTAL_700_dc_100set_top-k_k8/recon \
#   --patch_size 192 \
#   --kid_subset_size 1000 \
#   --kid_subsets 50

# python compute_patch_KID_final.py \
#   --label_dir /home/dongju1126/ICML_Triadic_Posterior/FlowDPS_Ours/workdir_phase_retrieval_demo_REBUTTAL_700_all_fixed_100set_top-k_k8/label \
#   --recon_dir /home/dongju1126/ICML_Triadic_Posterior/FlowDPS_Ours/workdir_phase_retrieval_demo_REBUTTAL_700_all_fixed_100set_top-k_k8/recon \
#   --patch_size 192 \
#   --kid_subset_size 1000 \
#   --kid_subsets 50

####### uniform sched. ######################
python compute_patch_KID_final.py \
  --label_dir /home/dongju1126/ICML_Triadic_Posterior/FlowDPS_Ours/workdir_uniform_fixed_REBUTTAL/GD_FFHQ_2/label \
  --recon_dir /home/dongju1126/ICML_Triadic_Posterior/FlowDPS_Ours/workdir_uniform_fixed_REBUTTAL/GD_FFHQ_2/recon \
  --patch_size 192 \
  --kid_subset_size 1000 \
  --kid_subsets 50