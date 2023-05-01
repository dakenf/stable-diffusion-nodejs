# Stable Diffusion for Node.js with GPU acceleration on Cuda or DirectML

## Info
This is a pure typescript implementation of SD pipeline that runs ONNX versions of the model with [patched ONNX node runtime](https://github.com/dakenf/onnxruntime-node-gpu)

## Requirements
Warning: this project requires Node 18

### Windows
Works out of the box with DirectML. No additional libraries required

You can speed up things by installing tfjs-node, but i haven't seen significant performance improvements https://github.com/tensorflow/tfjs/tree/master/tfjs-node

It might require installing visual studio build tools and python 2.7 https://community.chocolatey.org/packages/visualstudio2022buildtools

### Linux / WSL2
1. Install CUDA (tested only on 11-7 but 12 should be supported) https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
2. Install onnxruntime-linux-x64-gpu-1.14.1 https://github.com/microsoft/onnxruntime/releases/tag/v1.14.1
### Mac OS M1
No requirements but can run only on CPU which is quite slow (about 0.2s/it for fp32 and 0.1s/it for fp16)

## Usage
### Basic windows with SD 2.1
```typescript
import { PNG } from 'pngjs'
import { StableDiffusionPipeline } from 'stable-diffusion-nodejs'

const pipe = await StableDiffusionPipeline.fromPretrained(
  'directml', // can be 'cuda' on linux or 'cpu' on mac os
  'aislamov/stable-diffusion-2-1-base-onnx', // relative path or huggingface repo with onnx model
)

const image = await pipe.run("A photo of a cat", undefined, 1, 9, 30)
const p = new PNG({ width: 512, height: 512, inputColorType: 2 })
p.data = Buffer.from((await image[0].data()))
p.pack().pipe(fs.createWriteStream('output.png')).on('finish', () => {
  console.log('Image saved as output.png');
})
```

### Accelerated with tfjs-node SD 2.1
```typescript
import * as tf from "@tensorflow/tfjs-node"
import { StableDiffusionPipeline } from 'stable-diffusion-nodejs'

const pipe = await StableDiffusionPipeline.fromPretrained(
  'directml', // can be 'cuda' on linux or 'cpu' on mac os
  'aislamov/stable-diffusion-2-1-base-onnx', // relative path or huggingface repo with onnx model
)

const image = await pipe.run("A photo of a cat", undefined, 1, 9, 30)
const png = await tf.node.encodePng(image[0])
fs.writeFileSync("output.png", png);
```

### To run 1.X models you need to pass huggingface hub revision and version number = 1
```typescript
import * as tf from "@tensorflow/tfjs-node"
import { StableDiffusionPipeline } from 'stable-diffusion-nodejs'

const pipe = await StableDiffusionPipeline.fromPretrained(
  'directml', // can be 'cuda' on linux or 'cpu' on mac os
  'CompVis/stable-diffusion-v1-4',
  'onnx', // hf hub revision
  1, // SD version, cannot detect automatically yet
)

const image = await pipe.run("A photo of a cat", undefined, 1, 9, 30)
const png = await tf.node.encodePng(image[0])
fs.writeFileSync("output.png", png);
```

## Command-line usage
To test inference run this command. It will download SD2.1 onnx version from huggingface hub
### Windows
`npm run txt2img -- --prompt "an astronaut riding a horse" --provider directml`
### Linux
`npm run txt2img -- --prompt "an astronaut riding a horse" --provider cuda`

You can also use `--provider cpu` on a mac or if you don't have a supported videocard

## Converting other models to ONNX
You can use this tool to convert any HF hub model to ONNX https://github.com/Amblyopius/Stable-Diffusion-ONNX-FP16
Use fp16 for Cuda/DirectML and fp32 for Apple M1 (it runs twice faster but still slow)

## Roadmap
1. Support different schedulers, like DDIMS and UniPCMultistepScheduler
2. Support batch size > 1
3. ControlNet support
4. Add interop between ONNX backend and tensorflow.js to avoid copying data from and to GPU on each inference step
