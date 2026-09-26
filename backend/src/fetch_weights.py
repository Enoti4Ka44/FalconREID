"""Загрузка весов по манифесту с проверкой sha256.

Веса моделей (~770 МиБ) не помещаются в git: GitHub не принимает файлы
больше 100 МБ. Постановщик (Q&A, вопрос 39) разрешает скачивать веса во
время `docker build` — при условии, что источник зафиксирован и проверяется
контрольная сумма. Во время работы контейнера сеть не нужна.

Список файлов — `models/weights.json`:
    {"files": [{"name": "final_dinov3ps.pt", "url": "...", "sha256": "...",
                "bytes": 123}]}

    python -m src.fetch_weights            # скачать недостающие, проверить все
    python -m src.fetch_weights --make     # пересчитать sha256 по файлам в models/
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

MODELS = Path("models")
MANIFEST = MODELS / "weights.json"
RELEASE = "https://github.com/Misha-Hromov-32/MoscowHack/releases/download/{tag}/{name}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make(tag: str):
    """Собрать models/weights.json по файлам весов, лежащим в models/."""
    files = []
    for p in sorted(list(MODELS.glob("*.pt")) + list(MODELS.glob("*.npz"))):
        files.append({"name": p.name, "url": RELEASE.format(tag=tag, name=p.name),
                      "sha256": sha256(p), "bytes": p.stat().st_size})
        print(f"  {p.name:28s} {p.stat().st_size / 2**20:8.1f} МиБ  {files[-1]['sha256'][:16]}…")
    MANIFEST.write_text(json.dumps({"release": tag, "files": files}, indent=1,
                                   ensure_ascii=False), encoding="utf-8")
    print(f"saved: {MANIFEST} ({len(files)} файлов)")


def fetch():
    spec = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ok = True
    for f in spec["files"]:
        dst = MODELS / f["name"]
        if not dst.exists() or dst.stat().st_size != f["bytes"]:
            print(f"скачиваю {f['name']} ({f['bytes'] / 2**20:.0f} МиБ)…", flush=True)
            tmp = dst.with_suffix(dst.suffix + ".part")
            urllib.request.urlretrieve(f["url"], tmp)
            tmp.replace(dst)
        got = sha256(dst)
        if got != f["sha256"]:
            print(f"ОШИБКА контрольной суммы {f['name']}: {got} != {f['sha256']}")
            ok = False
        else:
            print(f"  {f['name']}: sha256 совпадает")
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--tag", default="weights-v4")
    a = ap.parse_args()
    make(a.tag) if a.make else fetch()


if __name__ == "__main__":
    main()
