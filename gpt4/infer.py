from pathlib import Path
import argparse
import ast
import asttokens
import asttokens.util
import token
import strip_hints
import autopep8
import openai
import asyncio
import traceback
import tqdm
from parse import parse_response, ParsingFailure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, default=Path("responses"))
    parser.add_argument("--api-key", "-k", type=str, default=None)
    parser.add_argument("--model", "-m", type=str, default="gpt-4-0314")
    parser.add_argument("--num-preds", "-np", type=int, default=20)
    parser.add_argument("--num-samples", "-ns", type=int, default=3)
    parser.add_argument("--temperature", "-temp", type=float, default=1)
    parser.add_argument(
        "--prompt",
        "-p",
        type=Path,
        default=(Path(__file__).parent / "prompt_template.txt"),
    )
    parser.add_argument("--max-iters", "-mi", type=int, default=3)
    parser.add_argument("--restart", "-r", action="store_true")
    parser.add_argument("--prompt-only", "-po", action="store_true")

    args = parser.parse_args()

    input_path: Path = args.input
    if input_path.is_dir():
        paths = list(input_path.glob("**/*.pydd"))
    else:
        paths = [input_path]

    openai.api_key = args.api_key

    if args.temperature == 0 and args.num_samples > 1:
        print("num_samples is set to 1 because temperature is 0")
        args.num_samples = 1

    prompt_template: str = args.prompt.read_text()

    async def infer_all():
        tasks = []
        name_to_idx = {}

        for path in paths:
            path = path.resolve()
            name = (path.parent.name, path.name)

            src = path.read_text()
            src = autopep8.fix_code(strip_type_hints(strip_comments(src)))

            if args.prompt_only:
                print(f"Prompt for {name}:")
                print("=" * 80)
                print(prompt_template.format(src=src, num_preds=args.num_preds))
                print("=" * 80)
                input("Press any key to continue...\n")
                continue

            output_dir = args.output
            (output_dir / name[0]).mkdir(parents=True, exist_ok=True)
            if args.restart:
                name_to_idx[name] = 0
            else:
                name_to_idx[name] = args.num_samples
                for idx in range(args.num_samples):
                    if not (output_dir / name[0] / (name[1] + f".{idx}.txt")).exists():
                        name_to_idx[name] = idx
                        break

            for _ in range(args.num_samples - name_to_idx[name]):
                tasks.append(
                    asyncio.create_task(
                        infer(
                            name,
                            src,
                            prompt_template,
                            args.model,
                            args.num_preds,
                            args.temperature,
                            args.max_iters,
                        )
                    )
                )

        for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            try:
                name, response = await task
            except Exception as e:
                print(f"Failed to infer: {repr(e)}")
                traceback.print_exc()
                continue
            idx = name_to_idx[name]
            name_to_idx[name] += 1
            print(name, idx)
            response_path = output_dir / name[0] / (name[1] + f".{idx}.txt")
            response_path.write_text(response)

    asyncio.run(infer_all())


async def infer(
    name: tuple[str, str],
    src: str,
    prompt_template: str,
    model: str,
    num_preds: int,
    temperature: float,
    max_iters: int,
):
    prompt = prompt_template.format(src=src, num_preds=num_preds)
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]
    response_content = ""
    success = False

    for _ in range(max_iters):
        # Use the streaming API to avoid timeout caused by long content
        content_chunks = []
        async for chunk in await openai.ChatCompletion.acreate(  # type: ignore
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        ):
            content_chunks.append(chunk.choices[0].delta.get("content", ""))

        content = "".join(content_chunks)
        response_content += content

        try:
            type_seqs = parse_response(response_content, src)
        except ParsingFailure:
            raise Exception(f"{name}: failed to parse response:\n{response_content}")
        if type_seqs is not None:
            success = True
            break

        messages.append(
            {
                "role": "assistant",
                "content": content,
            }
        )
        messages.append(
            {
                "role": "user",
                "content": "continue",
            },
        )

    if not success:
        raise Exception(f"{name}: max chat iterations exceeded")

    return name, response_content


def strip_comments(src: str):
    ast_tokens = asttokens.ASTTokens(src, parse=True)
    replacements = []
    for t in ast_tokens.tokens:
        if t.type == token.COMMENT or t.type == token.TYPE_COMMENT:
            replacements.append((t.startpos, t.endpos, ""))
    for node in asttokens.util.walk(ast_tokens.tree):
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Str):
                startpos, endpos = ast_tokens.get_text_range(node)
                replacements.append((startpos, endpos, ""))
    return asttokens.util.replace(src, replacements)


def strip_type_hints(src: str):
    return strip_hints.strip_string_to_string(src, to_empty=True, strip_nl=True)


if __name__ == "__main__":
    main()
