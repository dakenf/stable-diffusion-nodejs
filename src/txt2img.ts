import minimist from 'minimist';
import { StableDiffusionPipeline } from './StableDiffusionPipeline'
import fs from 'fs'
import { PNG } from 'pngjs'

interface CommandLineArgs {
  m: string;
  prompt: string;
  negativePrompt?: string;
  provider?: 'cuda'|'cpu'|'directml';
  rev?: string;
  version?: 1|2;
  steps: number
}

function parseCommandLineArgs(): CommandLineArgs {
  const args = minimist(process.argv.slice(2));

  return {
    m: args.m || 'aislamov/stable-diffusion-2-1-base-onnx',
    prompt: args.prompt || 'an astronaut riding a horse',
    negativePrompt: args.negativePrompt || '',
    provider: args.provider || 'cpu',
    rev: args.rev,
    version: args.version || 2,
    steps: args.steps || 30,
  }
}

async function main() {
  const args = parseCommandLineArgs();
  const pipe = await StableDiffusionPipeline.fromPretrained(
    args.provider,
    args.m,
    args.rev,
    args.version,
  )

  let image = await pipe.run(args.prompt, args.negativePrompt, 1, 9, args.steps)
  const p = new PNG({ width: 512, height: 512, inputColorType: 2 })
  p.data = Buffer.from((await image[0].data()))
  p.pack().pipe(fs.createWriteStream('output.png')).on('finish', () => {
    console.log('Image saved as output.png');
  })
}

main();
