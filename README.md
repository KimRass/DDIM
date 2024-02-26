# 1. Sampling
## 1) `"normal"` mode
- `trunc_normal_thresh=None`:
    - <img src="https://github.com/KimRass/KimRass/assets/67457712/e3e4731d-98cc-41de-b63b-9d3136bcbaed" width="700">
- `trunc_normal_thresh=0.1`:
    - <img src="https://github.com/KimRass/KimRass/assets/67457712/0cc7a638-996b-4ce1-b5c1-146e4249e962" width="276">
- `trunc_normal_thresh=0.5`:
    - <img src="https://github.com/KimRass/KimRass/assets/67457712/187e88b5-fc00-4e9e-bc32-3f9ceaec9d2c" width="276">
- `trunc_normal_thresh=1`:
    - <img src="https://github.com/KimRass/KimRass/assets/67457712/7104e158-28fe-43e2-8051-e8f6a49908eb" width="276">
- `trunc_normal_thresh=1.5`:
    - <img src="https://github.com/KimRass/KimRass/assets/67457712/0b6585d8-db1a-46e0-b85c-a9622d4f379d" width="276">
- `trunc_normal_thresh=2`:
    - <img src="https://github.com/KimRass/KimRass/assets/67457712/748f8e04-e5b2-4824-aaf2-a9fed04796a7" width="276">
- `trunc_normal_thresh=2.5`:
    - <img src="https://github.com/KimRass/KimRass/assets/67457712/89dcba0c-1779-42e2-9693-c6f12c735c02" width="276">
- `trunc_normal_thresh=3`:
    - <img src="https://github.com/KimRass/KimRass/assets/67457712/f2d4a0d6-9f29-459f-89ef-ef95f2e39f77" width="276">
## 2) `"interpolation"` mode
- <img src="https://github.com/KimRass/KimRass/assets/67457712/8403ce36-c1d0-4289-874d-1ca48cec8f9d" width="700">
## 3) `"interpolation_on_grid"` mode
- <img src="https://github.com/KimRass/KimRass/assets/67457712/127dc5a2-4b56-4d59-af74-460848aceb50" width="700">
- <img src="https://github.com/KimRass/KimRass/assets/67457712/40a5c5a9-379b-4fd8-9659-1dc933e48c3e" width="700">

# 2. Theoretical Background
- "Predicted $x_{0}$":
$$\frac{x_{t} - \sqrt{1 - \alpha_{t}}\epsilon_{\theta}^{(t)}(x_{t})}{\sqrt{\alpha_{t}}}$$
- "Direction pointing to $x_{t}$":
$$\sqrt{1 - \alpha_{t - 1} - \sigma_{t}^{2}} \cdot \epsilon_{\theta}^{(t)}(x_{t})$$
## 1) Backward (Denoising) Process
$$x_{t - 1} = \sqrt{\alpha_{t - 1}}\Bigg(\frac{x_{t} - \sqrt{1 - \alpha_{t}}\epsilon_{\theta}}{\sqrt{\alpha_{t}}}\Bigg) + \sqrt{1 - \alpha_{t - 1}}\epsilon_{\theta}$$