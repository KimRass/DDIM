# 1. Sampling
## 1) `"normal"` mode
- <img src="https://github.com/KimRass/KimRass/assets/67457712/e3e4731d-98cc-41de-b63b-9d3136bcbaed" width="700">
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