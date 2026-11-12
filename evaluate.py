"""Іздеу мен кеңесші сапасын тексеру / Оценка качества поиска и консультанта.

Іске қосу:  python evaluate.py
Метрикалар:
  Top-1  — дұрыс кітап бірінші орында тұрған сұраулар үлесі
  Top-3  — дұрыс кітап алғашқы үштікте тұрған сұраулар үлесі
  MRR    — дұрыс жауаптың орнына кері шаманың орташасы (1 = мінсіз)
  Intent — кеңесші сұрау түрін дұрыс анықтаған үлес
"""
import json
import time
from pathlib import Path

from core.assistant import Assistant
from core.search import SearchEngine, load_books


def main() -> None:
    tests = json.loads((Path(__file__).parent / "tests" / "test_queries.json").read_text(encoding="utf-8"))
    engine = SearchEngine(load_books())
    assistant = Assistant(engine)
    print(f"Іздеу режимі / Режим поиска: {engine.mode}\n")

    by_lang = {"kk": [], "ru": []}
    times = []
    for case in tests["search"]:
        t0 = time.perf_counter()
        ids = [r.book["id"] for r in engine.search(case["q"], top_k=5)]
        times.append(time.perf_counter() - t0)
        rank = next((i + 1 for i, b in enumerate(ids) if b in case["expected"]), None)
        by_lang[case["lang"]].append(rank)
        mark = "✓" if rank == 1 else ("~" if rank and rank <= 3 else "✗")
        print(f"{mark} [{case['lang']}] {case['q']:<45} rank={rank}")

    def report(name, ranks):
        n = len(ranks)
        top1 = sum(r == 1 for r in ranks) / n
        top3 = sum(r is not None and r <= 3 for r in ranks) / n
        mrr = sum(1 / r for r in ranks if r) / n
        print(f"{name:<12} n={n:<3} Top-1={top1:.0%}  Top-3={top3:.0%}  MRR={mrr:.2f}")

    print("\n=== Іздеу / Поиск ===")
    report("Қазақша", by_lang["kk"])
    report("Русский", by_lang["ru"])
    report("Барлығы", by_lang["kk"] + by_lang["ru"])
    hard = []
    print("\n--- Күрделі (перифраз) сұраулар / Сложные перефразированные запросы ---")
    for case in tests.get("hard", []):
        ids = [r.book["id"] for r in engine.search(case["q"], top_k=5)]
        rank = next((i + 1 for i, b in enumerate(ids) if b in case["expected"]), None)
        hard.append(rank)
        mark = "✓" if rank == 1 else ("~" if rank and rank <= 3 else "✗")
        print(f"{mark} [{case['lang']}] {case['q']:<45} rank={rank}")
    if hard:
        report("Күрделі", hard)
    print(f"Орташа жауап уақыты / Среднее время ответа: {1000 * sum(times) / len(times):.1f} мс")

    print("\n=== ЖИ-кеңесші: сұрау түрін анықтау / Определение намерения ===")
    ok = 0
    for case in tests["intent"]:
        got = assistant.answer(case["q"]).intent
        good = got == case["expected"]
        ok += good
        print(f"{'✓' if good else '✗'} {case['q']:<45} {got} (күтілген: {case['expected']})")
    print(f"Intent accuracy: {ok}/{len(tests['intent'])} = {ok / len(tests['intent']):.0%}")


if __name__ == "__main__":
    main()
