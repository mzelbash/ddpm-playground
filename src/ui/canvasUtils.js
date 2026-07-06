import * as tf from '@tensorflow/tfjs';

// Render a [-1,1] image tensor ([H,W,1] or [H,W]) into a canvas element.
// The rescale from [-1,1] to [0,1] is critical — forgetting it produces black/clipped output.
export async function tensorToCanvas(tensor, canvasEl) {
  const rescaled = tf.tidy(() => tensor.add(1).div(2).clipByValue(0, 1));
  await tf.browser.toPixels(rescaled, canvasEl);
  rescaled.dispose();
}
