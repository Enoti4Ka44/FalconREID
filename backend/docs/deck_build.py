# -*- coding: utf-8 -*-
"""Сборка презентации решения на базе шаблона ЛЦТ-2026.

  python docs/deck_build.py --base <deck_base2.pptx> --out <итог.pptx>

База — 14 слайдов шаблона в порядке: титул, о решении, проблема, путь,
ML, сервис, метрики, отказ, ошибки, скорость, демо, ТЗ, развитие, команда.
Числа берутся из artifacts/*.json и models/threshold.json (если есть).
Позиции выверены по рендеру шаблона: 100 px рендера = 1 дюйм.
"""
import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ACC = RGBColor(0xFF, 0x00, 0x53)
ACC2 = RGBColor(0x8A, 0x83, 0xD1)
PINK = RGBColor(0xFF, 0xD6, 0xE4)
DARKV = RGBColor(0x31, 0x0F, 0x53)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARKTXT = RGBColor(0x2A, 0x1A, 0x3A)
GREY = RGBColor(0x6A, 0x62, 0x80)
FONT = "Montserrat"

I = Inches


def load_numbers():
    n = {}
    def read(p):
        p = Path(p)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    rep = read("artifacts/val_report.json")
    if rep:
        r = rep["ranking"]
        n.update(mAP=f"{r['mAP']:.3f}", R1=f"{r['Rank-1']:.3f}",
                 R5=f"{r['Rank-5']:.3f}", mINP=f"{r['mINP']:.3f}")
    thr = read("models/threshold.json")
    if thr:
        n.update(thr=f"{thr['threshold']:.3f}", F1=f"{thr['val_F1']:.3f}",
                 TNR=f"{thr['val_TNR']:.3f}")
        if "val_precision" in thr:
            n.update(prec=f"{thr['val_precision']:.2f}", rec=f"{thr['val_recall']:.2f}")
    b = read("artifacts/benchmark.json")
    if b:
        n.update(lat=f"{b['latency_bs1']['p50_ms']:.0f}", fps=f"{b['throughput']['fps']:.0f}")
    a = read("artifacts/ann_demo.json")
    if a:
        n.update(ann_ms=f"{a['hnsw_ms_per_query']:.1f}",
                 ann_recall=f"{a['recall@10_vs_exact']:.2f}")
    return n


def txt(slide, x, y, w, h, parts, size=13, color=DARKTXT, bold=False,
        align=PP_ALIGN.LEFT, spacing=1.12, anchor=None):
    """parts: строка или список (текст, {опции}) построчно."""
    box = slide.shapes.add_textbox(I(x), I(y), I(w), I(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if anchor:
        tf.vertical_anchor = anchor
    if isinstance(parts, str):
        parts = [(parts, {})]
    first = True
    for text, opt in parts:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = opt.get("align", align)
        p.line_spacing = opt.get("spacing", spacing)
        p.space_after = Pt(opt.get("after", 4))
        run = p.add_run()
        run.text = text
        f = run.font
        f.name = FONT
        f.size = Pt(opt.get("size", size))
        f.bold = opt.get("bold", bold)
        f.color.rgb = opt.get("color", color)
    return box


def chip_title(slide, text, light=False):
    """Заголовок в фирменном чипе слева сверху (чип уже есть в макете)."""
    txt(slide, 0.42, 0.36, 3.32, 0.62, text, size=17, bold=True,
        color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def pic(slide, path, x, y, w=None, h=None):
    p = Path(path)
    if p.exists():
        return slide.shapes.add_picture(str(p), I(x), I(y),
                                        I(w) if w else None, I(h) if h else None)


def bullet_list(slide, x, y, w, items, size=12.5, gap=6, color=DARKTXT,
                mark_color=ACC):
    parts = []
    for it in items:
        parts.append((f"▪  {it}", {"size": size, "color": color, "after": gap}))
    txt(slide, x, y, w, 5.5, parts)


def iter_shapes(shapes):
    """Рекурсивный обход шейпов, включая группы."""
    for shape in shapes:
        try:
            sub = shape.shapes  # только у групп
        except AttributeError:
            yield shape
        else:
            yield from iter_shapes(sub)


def replace_text_runs(slide, mapping):
    """Замена текста в существующих плейсхолдерах шаблона (включая группы)."""
    for shape in iter_shapes(slide.shapes):
        if not shape.has_text_frame:
            continue
        for para in shape.text_frame.paragraphs:
            joined = "".join(r.text for r in para.runs)
            for old, new in mapping.items():
                if old in joined and para.runs:
                    para.runs[0].text = new
                    for r in para.runs[1:]:
                        r.text = ""
                    break


def delete_ghosts(slide, needle="Образец текста"):
    """Удалить бледные подписи-заглушки — и со слайда, и с его макета."""
    for holder in (slide, slide.slide_layout):
        for shape in list(iter_shapes(holder.shapes)):
            if shape.has_text_frame and needle in shape.text_frame.text:
                shape._element.getparent().remove(shape._element)


def delete_charts(slide):
    """Удалить графики-образцы с макета (на их место встают наши PNG)."""
    for shape in list(slide.shapes):
        if shape.shape_type == 3 or shape.has_chart if hasattr(shape, "has_chart") else False:
            pass
    # прямое удаление graphicFrame
    for el in list(slide.shapes._spTree):
        if el.tag.endswith("graphicFrame"):
            slide.shapes._spTree.remove(el)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figures", default="artifacts/figures")
    ap.add_argument("--shots", default="artifacts/screenshots")
    a = ap.parse_args()
    n = load_numbers()
    g = lambda k, d="—": str(n.get(k, d))
    FIG, SHOT = a.figures, a.shots

    prs = Presentation(a.base)
    s = prs.slides

    # ---------- 0. Титул ----------
    sl = s[0]
    txt(sl, 0.55, 2.1, 6.0, 2.6, [
        ("ФАЛЬКОН·ReID", {"size": 44, "bold": True, "color": WHITE, "after": 8}),
        ("Цифровой признак транспортного средства", {"size": 20, "color": WHITE, "after": 4}),
        ("Поиск автомобиля по внешности — без государственного номера",
         {"size": 14, "color": PINK, "after": 0}),
    ])
    txt(sl, 0.55, 5.9, 6.5, 1.2, [
        ("Кейс №7 · «Фалькон Тех» · ЛЦТ-2026", {"size": 13, "bold": True, "color": WHITE, "after": 2}),
        ("Команда: заполните название и состав", {"size": 11.5, "color": PINK}),
    ])

    # ---------- 1. Коротко о решении ----------
    sl = s[1]
    replace_text_runs(sl, {
        "Опишите в чем техническая составляющая вашего решения":
            "Четверной ансамбль DINOv2 ViT-L@336 + ViT-B с частями + ConvNeXt + DINOv3-L; "
            "CosFace + batch-hard triplet; постобработка DBA/alphaQE. Эмбеддинг "
            "3840-d, косинусный поиск FAISS. FastAPI + OpenAPI, веб-клиент, Docker офлайн.",
        "Опишите ваши идеи по дальнейшему применению, развитию или внедрению":
            "Оператор находит машину по снимку за секунды вместо часов ручного "
            "просмотра архива. Честный отказ вместо ложных находок. "
            "Порог уверенности управляется без переобучения.",
    })
    bullet_list(sl, 0.72, 3.6, 5.3, [
        f"mAP {g('mAP')} · Rank-1 {g('R1')} · Rank-5 {g('R5')} на незнакомых ТС",
        "4 больших модели в fp16 — 1505 МБ из лимита 2 ГБ",
        "Запуск одной командой: docker compose up, интернет не нужен",
        "Интерпретируемость: карта внимания к решению модели",
    ], size=12)
    bullet_list(sl, 7.15, 3.6, 5.3, [
        f"Режим отказа: F1 {g('F1')} при TNR {g('TNR')} — порог обоснован",
        "Сами восстанавливаем камеры по фону кадра — отсев дублей сцены",
        "Открытый стек: PyTorch, FastAPI, FAISS (BSD/MIT/Apache)",
        "Готов к пилоту на городском потоке",
    ], size=12)

    # ---------- 2. Проблема ----------
    sl = s[2]
    txt(sl, 0.75, 1.75, 11.8, 0.75,
        "Номер не читается — город теряет машину",
        size=26, bold=True, color=DARKTXT)
    cols = [
        ("Грязь и снег", "Регистрационный знак нечитаем в плохую погоду — ALPR бессилен."),
        ("Блики и ночь", "Свет фар и солнца засвечивает знак; ночью деталей мало."),
        ("Перекрытие и ракурс", "Знак закрыт другим авто или вовсе не в поле зрения камеры."),
        ("Ручной поиск", "Оператор листает архив сотни часов — и часто безрезультатно."),
    ]
    for i, (h, t) in enumerate(cols):
        x = 0.75 + i * 3.0
        txt(sl, x, 2.85, 2.7, 0.5, h, size=15, bold=True, color=ACC)
        txt(sl, x, 3.35, 2.7, 1.6, t, size=11.5, color=DARKTXT)
    txt(sl, 0.75, 5.35, 11.8, 1.2, [
        ("Решение: устойчивый цифровой признак по внешности — цвет, геометрия, "
         "оптика, наклейки, повреждения.", {"size": 15, "bold": True, "color": DARKV, "after": 3}),
        ("Снимки одного ТС с разных камер сходятся в векторном пространстве — "
         "без какой-либо привязки к номеру.", {"size": 12.5, "color": GREY}),
    ])

    # ---------- 3. Путь оператора (таймлайн 1-5) ----------
    sl = s[3]
    chip_title(sl, "ПУТЬ ОПЕРАТОРА")
    delete_ghosts(sl)
    steps_top = [
        (0.30, 1.55, 3.05, "Загрузка кадра",
         "Снимок с камеры в браузере или по API. BBox — из разметки или мышью прямо на кадре."),
        (5.15, 1.55, 3.05, "Цифровой признак",
         "Четверной ансамбль формирует вектор 3840-d за ~0.17 с. Только внешность — номер не используется."),
        (10.00, 1.55, 3.00, "Поиск по галерее",
         "Косинусная близость ко всему архиву. До 10⁶ снимков — FAISS HNSW за миллисекунды."),
    ]
    steps_bottom = [
        (2.75, 5.35, 3.05, "Решение сервиса",
         "Топ-N кандидатов с уверенностью либо честный отказ, если совпадения нет."),
        (7.65, 5.35, 3.05, "Работа с результатом",
         "Карта внимания «почему совпало», экспорт CSV, передача материалов в работу."),
    ]
    for x, y, w, h, t in [(*st,) for st in steps_top] + [(*st,) for st in steps_bottom]:
        txt(sl, x + 0.15, y + 0.12, w - 0.3, 0.4, h, size=13.5, bold=True, color=DARKV)
        txt(sl, x + 0.15, y + 0.55, w - 0.3, 1.15, t, size=10.5, color=DARKTXT)
    # подписи «Образец текста» затираем белыми накладками не требуется: они бледные

    # ---------- 4. Как работает ML ----------
    sl = s[4]
    chip_title(sl, "КАК РАБОТАЕТ ML")
    heads = [(1.20, "Признак"), (5.48, "Обучение"), (9.85, "Инвариантность")]
    for x, h in heads:
        txt(sl, x, 3.02, 2.4, 0.4, h, size=13, bold=True, color=WHITE,
            align=PP_ALIGN.CENTER)
    cards = [
        (0.60, 3.45, 3.95, [
            ("Четверной ансамбль", {"bold": True, "size": 12.5, "color": DARKV, "after": 3}),
            ("DINOv2 ViT-L + ViT-B, DINOv3-L (два поколения самообучения "
             "Meta) + ConvNeXt-base (свёртка): признаки декоррелируют "
             "ошибки — +4.9 mAP к лучшей одиночной модели.", {"size": 11, "after": 6}),
            ("BNNeck → конкатенация 1.0/0.7/0.5/1.0 → 3840-d float32; "
             "постобработка DBA k=3 + alphaQE k=1 даёт ещё +3.4 mAP.", {"size": 11}),
        ]),
        (4.88, 3.45, 3.95, [
            ("CosFace + batch-hard triplet", {"bold": True, "size": 12.5, "color": DARKV, "after": 3}),
            ("Классификация 1541 идентичности с угловым отступом (s=48, m=0.25) + "
             "триплеты по самым трудным парам в батче.", {"size": 11, "after": 6}),
            ("PK-сэмплирование 16×4: в каждом батче — разные ракурсы одного ТС.",
             {"size": 11}),
        ]),
        (9.15, 3.45, 3.95, [
            ("Ракурс, свет, перекрытия", {"bold": True, "size": 12.5, "color": DARKV, "after": 3}),
            ("Аугментации: отражение, color jitter, blur, random erasing "
             "(имитация перекрытий).", {"size": 11, "after": 6}),
            ("TTA-отражение на инференсе. Каждый train-ТС снят с 2+ камер — "
             "модель учится сближать ракурсы.", {"size": 11}),
        ]),
    ]
    for x, y, w, parts in cards:
        txt(sl, x + 0.22, y + 0.18, w - 0.44, 3.6, parts)

    # ---------- 5. Архитектура сервиса ----------
    sl = s[5]
    chip_title(sl, "АРХИТЕКТУРА")
    txt(sl, 0.62, 1.72, 5.5, 0.4, "Поток обработки запроса",
        size=14, bold=True, color=DARKV)
    flow = [
        ("1 · Получение", "изображение + BBox через API/UI, валидация входа"),
        ("2 · Обработка", "кроп +6% → четверной ансамбль (DINOv2 + DINOv3 + ConvNeXt) → вектор 3840-d"),
        ("3 · Анализ", "постобработка DBA/AQE → косинусный поиск FAISS"),
        ("4 · Результат", "топ-N с уверенностью или отказ при max sim < τ"),
    ]
    yy = 2.25
    for h, t in flow:
        txt(sl, 0.62, yy, 5.5, 0.35, h, size=12.5, bold=True, color=ACC)
        txt(sl, 0.62, yy + 0.34, 5.5, 0.6, t, size=11, color=DARKTXT)
        yy += 1.06
    right = [
        ("Тонкий клиент", "SPA в браузере: BBox мышью, порог на лету, экспорт CSV"),
        ("Backend инференса", "Python · FastAPI · OpenAPI (Swagger) · GPU/CPU"),
        ("Векторный индекс", "числовой индекс галереи; путь роста — pgvector/Milvus"),
        ("Контейнеризация", "Docker compose, одна команда, офлайн — веса в образе"),
        ("Интерпретируемость", "attention-карты: куда смотрела модель"),
    ]
    yy = 1.55
    for h, t in right:
        txt(sl, 6.72, yy + 0.10, 6.2, 0.35, h, size=12, bold=True, color=DARKV)
        txt(sl, 6.72, yy + 0.42, 6.2, 0.5, t, size=10.5, color=DARKTXT)
        yy += 1.105

    # ---------- 6. Метрики ----------
    sl = s[6]
    chip_title(sl, "МЕТРИКИ КАЧЕСТВА")
    delete_charts(sl)
    pic(sl, f"{FIG}/training_curve.png", 0.35, 2.15, 6.2)
    txt(sl, 0.35, 5.55, 6.2, 0.5,
        "Кривая обучения: лоссы и валидационный mAP",
        size=10.5, color=PINK, align=PP_ALIGN.CENTER)
    txt(sl, 0.35, 6.15, 6.2, 0.7,
        "Валидация: 308 незнакомых ТС · 1317 запросов ·\n35% запросов-дистракторов без пары в галерее",
        size=10.5, color=WHITE, align=PP_ALIGN.CENTER)
    mcards = [
        (1.15, [("mAP", ACC), (g("mAP"), None)],
         "кросс-камерный, основная метрика кейса"),
        (3.25, [("Rank-1 / Rank-5", ACC), (f"{g('R1')} / {g('R5')}", None)],
         "верный кандидат первым / в пятёрке"),
        (5.25, [("mINP", ACC), (g("mINP"), None)],
         "качество полного списка совпадений"),
    ]
    yy = [1.10, 3.20, 5.20]
    for (y, kv, note) in mcards:
        txt(sl, 7.00, y + 0.16, 5.9, 0.42, kv[0][0], size=13, bold=True, color=DARKV)
        txt(sl, 7.00, y + 0.55, 5.9, 0.75, kv[1][0], size=30, bold=True, color=ACC)
        txt(sl, 7.00, y + 1.28, 5.9, 0.5, note, size=10.5, color=GREY)

    # ---------- 7. Режим отказа ----------
    sl = s[7]
    chip_title(sl, "РЕЖИМ ОТКАЗА")
    delete_charts(sl)
    delete_ghosts(sl)
    pic(sl, f"{FIG}/similarity_hist.png", 0.50, 1.50, 5.25)
    pic(sl, f"{FIG}/threshold_curves.png", 0.50, 4.48, 5.25)
    rl = [
        (f"Порог τ = {g('thr')}",
         "максимум F1 при ограничении TNR ≥ 0.70 — сетка из 500 порогов на валидации"),
        (f"F1 = {g('F1')} · TNR = {g('TNR')}",
         "баланс находок и честных отказов; безусловный max F1 сидит при TNR ≈ 0.3 — недопустимо"),
        ("Ловушка теста: дубли той же сцены",
         "72% запросов имели «совпадение» с той же камеры: соседний кадр или та же припаркованная "
         "машина. Камеры-группы восстанавливаем по фону кадра и геометрии BBox (точность 0.97–1.0 "
         "на train) и исключаем одно-сценных кандидатов"),
        ("Порог — параметр продукта",
         "оператор меняет строгость в UI/API без переобучения; τ по умолчанию обоснован"),
    ]
    yy = 1.30
    for h, t in rl:
        txt(sl, 6.35, yy, 6.55, 0.4, h, size=13, bold=True, color=ACC)
        txt(sl, 6.35, yy + 0.40, 6.55, 0.85, t, size=10.5, color=DARKTXT)
        yy += 1.39

    # ---------- 8. Анализ ошибок ----------
    sl = s[8]
    txt(sl, 0.70, 0.55, 8.0, 0.6, "АНАЛИЗ ОШИБОК", size=24, bold=True, color=DARKV)
    ec = [
        (0.69, "«Близнецы»",
         "Одинаковая массовая модель, цвет и комплектация. Разделяются только "
         "микродеталями: диски, наклейки, повреждения.",
         "План: части-признаки и агрегация нескольких кадров трека.",
         f"{FIG}/errors_false_accept.png"),
        (4.84, "Разворот 180°",
         "Запрос «в лоб», в галерее — только корма. Совпадение находится, но "
         "уверенность падает к порогу.",
         "План: синтетика ракурсов и парные аугментации.",
         f"{FIG}/errors_rank1miss.png"),
        (9.18, "Ночь и дальний план",
         "Деталей физически мало — уверенность честно низкая, запрос уходит в отказ.",
         "Это корректное поведение: сервис знает, чего не знает.",
         f"{FIG}/hard_correct.png"),
    ]
    for x, h, t1, t2, img in ec:
        txt(sl, x + 0.25, 1.92, 3.05, 0.6, h, size=15, bold=True, color=ACC)
        p = pic(sl, img, x + 0.25, 2.55, 3.05)
        txt(sl, x + 0.25, 4.00, 3.05, 1.7, t1, size=10.5, color=DARKTXT)
        txt(sl, x + 0.25, 5.70, 3.05, 1.0, t2, size=10.5, bold=True, color=DARKV)

    # ---------- 9. Скорость и масштаб ----------
    sl = s[9]
    chip_title(sl, "СКОРОСТЬ И МАСШТАБ")
    perf = [
        (0.40, g("lat") + " мс", "признак одного ТС, батч = 1, включая препроцессинг"),
        (3.59, g("fps") + " FPS", "пакетная обработка на RTX 3060 без роста памяти"),
        (6.81, "1505 МБ", "4 модели в fp16 — в лимит 2048 МБ с запасом"),
        (10.04, "10⁶ галерея", f"FAISS HNSW: {g('ann_ms')} мс/запрос, recall@10 {g('ann_recall')}"),
    ]
    for x, big, note in perf:
        txt(sl, x + 0.25, 2.05, 2.4, 0.9, big, size=26, bold=True, color=ACC)
        txt(sl, x + 0.25, 3.05, 2.4, 2.2, note, size=11.5, color=DARKTXT)
    txt(sl, 0.65, 6.90, 12.0, 0.5,
        "Замеры: src/benchmark.py и src/ann_demo.py · официальные замеры — эталонный скрипт организаторов",
        size=10, color=PINK)

    # ---------- 10. Демо ----------
    sl = s[10]
    chip_title(sl, "ДЕМО")
    replace_text_runs(sl, {"lider": "localhost:8000", "www.": "", ".com": ""})
    pic(sl, f"{SHOT}/ui_results.png", 1.30, 1.72, 4.62)
    pic(sl, f"{SHOT}/ui_landing.png", 7.47, 1.72, 4.62)
    txt(sl, 1.30, 4.92, 4.62, 0.4, "Консоль оператора (стиль Linear)",
        size=12.5, bold=True, color=DARKV, align=PP_ALIGN.CENTER)
    txt(sl, 7.47, 4.92, 4.62, 0.4, "Лендинг сервиса · Swagger на /docs",
        size=12.5, bold=True, color=DARKV, align=PP_ALIGN.CENTER)
    feats = [
        "Рамка BBox рисуется мышью прямо на кадре",
        "Порог отказа и top-N меняются на лету",
        "Карта внимания: почему пары считаются совпадением",
        "Экспорт кандидатов в CSV одним кликом",
    ]
    for i, f in enumerate(feats):
        txt(sl, 1.30 + (i % 2) * 6.17, 5.62 + (i // 2) * 0.72, 4.9, 0.65,
            "▪  " + f, size=11.5, color=DARKTXT)

    # ---------- 11. Соответствие ТЗ ----------
    sl = s[11]
    txt(sl, 0.70, 0.55, 9.0, 0.6, "СООТВЕТСТВИЕ ТЗ И АРТЕФАКТЫ",
        size=24, bold=True, color=DARKV)
    rows = [
        ("Артефакты сдачи", "submission.csv · embeddings.npy · candidates.csv — формат и порядок проверены самотестом"),
        ("Запуск одной командой", "docker compose up → сервис+UI · compose run inference → артефакты; офлайн, веса в образе"),
        ("API и тонкий клиент", "OpenAPI/Swagger на /docs; доступ из любого браузера без установки ПО"),
        ("Ограничения ресурсов", "веса 1505 МБ ≤ 2048 МБ · инференс без OOM · латентность и FPS в benchmark.json"),
        ("Чистота решения", "признаки номера не используются (номера размыты; подтверждено attention-картами)"),
    ]
    ys = [1.82, 2.78, 3.74, 4.70, 5.62]
    for (h, t), y in zip(rows, ys):
        txt(sl, 0.70, y, 3.6, 0.8, "✓  " + h, size=13.5, bold=True, color=ACC)
        txt(sl, 4.55, y + 0.02, 8.3, 0.85, t, size=11.5, color=DARKTXT)

    # ---------- 12. Развитие ----------
    sl = s[12]
    chip_title(sl, "РАЗВИТИЕ")
    txt(sl, 0.62, 1.35, 6.9, 0.5, "Ближайшие шаги", size=14, bold=True, color=DARKV)
    txt(sl, 0.62, 1.80, 6.9, 1.4, [
        ("Части-признаки для «близнецов» · мультикадровая агрегация треков · "
         "калибровка уверенности на большем валидационном наборе",
         {"size": 12, "color": DARKTXT}),
    ])
    road = [
        ("Пилот", "интеграция с потоком камерного парка, dead-letter очереди, метрики качества на живых данных"),
        ("Масштаб", "pgvector/Milvus, шардирование по округам и времени, реплики API за балансировщиком"),
        ("Качество", "дообучение на потоке (semi-supervised), hard-negative mining на городских «близнецах»"),
        ("Продукт", "трекинг + мультикамерные треки, поиск по нескольким снимкам, роли и аудит операторов"),
    ]
    yy = 3.60
    for h, t in road:
        txt(sl, 0.62, yy, 2.1, 0.4, h, size=13.5, bold=True, color=ACC)
        txt(sl, 2.90, yy + 0.02, 9.9, 0.75, t, size=11.5, color=DARKTXT)
        yy += 0.86

    # ---------- 13. Команда ----------
    sl = s[13]
    replace_text_runs(sl, {
        "В чем суть вашего решения":
            "Сервис Re-ID: поиск автомобиля по внешности без номера. DINOv2 + "
            "CosFace, честный режим отказа, Docker офлайн, веб-интерфейс.",
        "Что делает ваше решение уникальным или инновационным?":
            "Обоснованный порог отказа (F1/TNR), интерпретируемость решений, "
            "готовность к 10⁶ галерее и запуск одной командой.",
    })

    prs.save(a.out)
    print("saved:", a.out)


if __name__ == "__main__":
    main()
