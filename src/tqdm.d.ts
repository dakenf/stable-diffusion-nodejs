export as namespace tqdm;



declare function tqdm (iterable: Iterable<any>, options?: Tqdm.Options): Iterable<any>

declare namespace Tqdm {
  interface Options {
    title?: string;
    total?: number;
    barFormat?: string;
    barLength?: number;
    barCompleteChar?: string;
    barIncompleteChar?: string;
  }
  function tqdm (iterable: Iterable<any>, options?: Tqdm.Options): Iterable<any>
}
export = tqdm;