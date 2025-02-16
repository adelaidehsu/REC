# Rate, Explain and Cite (REC): Enhanced Explanation and Attribution in Automatic Evaluation by Large Language Models

## Rec-Data
Our curated citations and explanation fine-tuning data is provided as `REC_data.zip`, which contains 24,081 (prompt, completion) pairs. Note that the full training data of the REC models additionally included [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2), [Skywork](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.1), [OffsetBias](https://huggingface.co/datasets/NCSOFT/offsetbias), and [Code Preference](https://huggingface.co/datasets/Vezora/Code-Preference-Pairs). The additional data can be downloaded at their own websites.

## Model training
If you wish to start the training from scratch -- the SFT trainer scripts for both REC-12B and REC-70B can be found at the `training` folder. Please put the unzipped REC_data at the correct path as you specify in the file, and modify the `output_dir` to be where you wish the trained model to be saved.
```
