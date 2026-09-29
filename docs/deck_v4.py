# -*- coding: utf-8 -*-
"""Пересборка презентации под продовую модель v4 на базе шаблона ЛЦТ-2026.

  python docs/deck_v4.py --deck <текущая презентация.pptx> \
      --template "<ЛЦТ2026 Шаблон презентации.pptx>" --shots <каталог скриншотов> \
      --out docs/ЛЦТ2026_ФальконТех_презентация.pptx

Что делает:
* добавляет обязательные слайды шаблона 9 («Участники») и 10 («История
  команды») в исходном дизайне и выстраивает обязательный блок в порядке
  шаблона 7–11: титул → о команде → участники → история → коротко о решении;
* заменяет все числа прежнего ансамбля v2 на метрики v4 (источники —
  models/threshold.json, artifacts/bench_official*.json, docs/PROGRESS.md);
* строит нативные диаграммы PowerPoint вместо устаревших картинок v2:
  рост mAP@10 по версиям и кривые F1 / TNR / балла по порогу τ;
* обновляет скриншоты интерфейса (React-клиент FalconREID).

--deck — презентация ДО пересборки (версия v2 из истории git:
`git show df95cd7~1:backend/docs/<имя>.pptx > deck_v2.pptx`); повторный прогон
на уже пересобранной добавил бы слайды шаблона второй раз.
"""
import argparse
import copy
import io
import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ACC = RGBColor(0xFF, 0x00, 0x53)
ACC2 = RGBColor(0x8A, 0x83, 0xD1)
DARKV = RGBColor(0x31, 0x0F, 0x53)
PINK = RGBColor(0xFF, 0xD6, 0xE4)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARKTXT = RGBColor(0x2A, 0x1A, 0x3A)
FONT = "Montserrat"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
ROOT = Path(__file__).resolve().parents[1] / "backend"


# ---------------------------------------------------------------- утилиты

def shape(slide, name, nth=0):
    found = [s for s in slide.shapes if s.name == name]
    if len(found) <= nth:
        raise KeyError(f"нет фигуры {name!r} на слайде")
    return found[nth]


def set_paras(shp, texts):
    """Заменяет текст абзацев, сохраняя форматирование первого run каждого абзаца.
    Лишние абзацы удаляются, недостающие клонируются с последнего."""
    tf = shp.text_frame
    paras = list(tf.paragraphs)
    while len(paras) < len(texts):
        new = copy.deepcopy(paras[-1]._p)
        paras[-1]._p.addnext(new)
        paras = list(tf.paragraphs)
    for p, t in zip(paras, texts):
        runs = p.runs
        if not runs:
            p.add_run().text = t
            continue
        runs[0].text = t
        for r in runs[1:]:
            r._r.getparent().remove(r._r)
    for p in paras[len(texts):]:
        p._p.getparent().remove(p._p)


def set_runs(shp, para_idx, texts):
    """Замена текста по run'ам внутри одного абзаца (для «Капитан: …» и т.п.)."""
    runs = shp.text_frame.paragraphs[para_idx].runs
    for r, t in zip(runs, texts):
        r.text = t


def copy_slide(src_slide, dst_prs, layout_name):
    """Перенос слайда шаблона (фон, фигуры, картинки) в другую презентацию
    с тем же макетом. Картинки копируются, r:embed переназначаются."""
    layout = next(l for l in dst_prs.slide_layouts if l.name == layout_name)
    new = dst_prs.slides.add_slide(layout)
    tree = new.shapes._spTree
    for el in list(tree):
        if el.tag not in (qn("p:nvGrpSpPr"), qn("p:grpSpPr")):
            tree.remove(el)
    rid_map = {}
    for rid, rel in src_slide.part.rels.items():
        if "image" in rel.reltype:
            _, new_rid = new.part.get_or_add_image_part(io.BytesIO(rel.target_part.blob))
            rid_map[rid] = new_rid

    def remap(el):
        for node in el.iter():
            for attr in (f"{{{R_NS}}}embed", f"{{{R_NS}}}link"):
                if node.get(attr) in rid_map:
                    node.set(attr, rid_map[node.get(attr)])
        return el

    src_csld = src_slide._element.cSld
    if src_csld.bg is not None:
        new._element.cSld.insert(0, remap(copy.deepcopy(src_csld.bg)))
    for el in src_slide.shapes._spTree:
        if el.tag in (qn("p:nvGrpSpPr"), qn("p:grpSpPr")):
            continue
        tree.append(remap(copy.deepcopy(el)))
    return new


def reorder(prs, order):
    lst = prs.slides._sldIdLst
    ids = list(lst)
    for el in ids:
        lst.remove(el)
    for i in order:
        lst.append(ids[i])


def replace_picture(slide, pic, image_path, width=None):
    """Новая картинка на месте старой (та же позиция и z-порядок)."""
    left, top = pic.left, pic.top
    w = Emu(width) if width else pic.width
    new = slide.shapes.add_picture(str(image_path), left, top, width=w)
    pic._element.addprevious(new._element)
    pic._element.getparent().remove(pic._element)
    return new


def style_chart(chart, color=DARKTXT, size=10):
    chart.font.name = FONT
    chart.font.size = Pt(size)
    chart.font.color.rgb = color


# ---------------------------------------------------------------- данные

def add_test_errors_slide(prs, figure):
    """Слайд с примерами из ТЕСТОВОЙ выборки (ТЗ §11): реальные ошибки и отказ v4."""
    layout = next(l for l in prs.slide_layouts if l.name == "Заголовок и объект")
    s = prs.slides.add_slide(layout)
    for ph in list(s.placeholders):
        if "Номер слайда" not in ph.name:
            ph._element.getparent().remove(ph._element)
    card = s.shapes.add_shape(5, Inches(0.36), Inches(1.20), Inches(12.62), Inches(5.95))  # скруглённый
    card.adjustments[0] = 0.04
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE
    card.line.fill.background()
    title = s.shapes.add_textbox(Inches(0.55), Inches(0.35), Inches(11.5), Inches(0.7))
    r = title.text_frame.paragraphs[0].add_run()
    r.text = "ОШИБКИ НА ТЕСТОВОЙ ВЫБОРКЕ"
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(24), True, WHITE
    s.shapes.add_picture(str(figure), Inches(0.55), Inches(1.40), height=Inches(5.55))
    notes = [
        ("Редкий класс", "автобусы — ~2% кадров train (оценка YOLOv8n) — чужие проходят порог. "
                         "Дальше: дообучение на автобусах и грузовиках."),
        ("Одна марка и модель", "верный кадр первый, хвост списка шумный. F1 считается по "
                                "top-1; хвост оператору — с относительным порогом."),
        ("Близнецы у порога", "уверенность ≈ τ: показываем оба кадра, решает оператор."),
        ("Отказ", "та же модель другого цвета — ниже τ, ложной находки нет."),
        ("Как отобраны", "src/make_test_examples.py по embeddings.npy и "
                         "test_cameras.csv; разметки теста нет, вывод — визуальный."),
    ]
    box = s.shapes.add_textbox(Inches(9.75), Inches(1.45), Inches(3.05), Inches(5.5))
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for head, body in notes:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(7)
        r1 = p.add_run()
        r1.text = head + ": "
        r1.font.name, r1.font.size, r1.font.bold, r1.font.color.rgb = FONT, Pt(10.5), True, ACC
        r2 = p.add_run()
        r2.text = body
        r2.font.name, r2.font.size, r2.font.color.rgb = FONT, Pt(10.5), DARKTXT
    return s


def numbers():
    thr = json.loads((ROOT / "models/threshold.json").read_text(encoding="utf-8"))
    bench = json.loads((ROOT / "artifacts/bench_official.json").read_text(encoding="utf-8"))
    return {
        "tau": thr["threshold"], "F1": thr["heldout_F1"], "TNR": thr["heldout_TNR"],
        "score": thr["heldout_score"], "curve": thr["curve_heldout"],
        "lat": bench["latency_b1_ms"], "fps": bench["best_fps"],
        # docs/PROGRESS.md, «Итог на 26 сентября» (30 переразбиений протокола)
        "map10": 0.8585, "r1": 0.823, "r5": 0.917,
        "map_hist": [("v2: 4 модели", 0.8268), ("v3: 2 модели", 0.8508),
                     ("v4 (прод)", 0.8585), ("ViT-H+ (архив)", 0.8828)],
    }


# ---------------------------------------------------------------- слайды

def build(deck, template, shots, out):
    prs = Presentation(deck)
    tpl = Presentation(template)
    n = numbers()
    S = list(prs.slides)          # исходный порядок: 0 титул … 13 о команде

    # обязательные слайды шаблона 9 и 10 — в исходном дизайне
    s_team = copy_slide(tpl.slides[8], prs, "Команда")
    s_hist = copy_slide(tpl.slides[9], prs, "Команда")

    # --- 1. титул
    set_paras(shape(S[0], "TextBox 6"),
              ["Кейс №7 · «Фалькон Тех» · ЛЦТ-2026", "Команда: [название команды]"])

    # --- 2. о команде (шаблон 8): решение и уникальность заполнены, данные команды — поля
    about = S[13]
    set_paras(shape(about, "Текст 8", 5),
              ["Сервис ReID находит тот же автомобиль на снимках других камер по внешности — "
               "без госномера. Две модели DINOv3-L + DINOv2-B, честный режим отказа, "
               "веб-интерфейс, запуск одной командой."])
    set_paras(shape(about, "Текст 8", 1),
              ["Порог отказа выбран по формуле балла постановщика; все улучшения доказаны "
               "парными замерами на 30 переразбиениях; 126 FPS при 770 МиБ весов; карта "
               "внимания показывает, что номер не используется."])

    # --- 3. участники: заголовок в чипе, карточки — поля шаблона для заполнения
    for t in (s_team, s_hist):
        for ph in t.placeholders:
            if ph.placeholder_format.type is not None and "Заголовок" in ph.name:
                ph.text_frame.text = "КОМАНДА" if t is s_team else "ИСТОРИЯ КОМАНДЫ"
                for r in ph.text_frame.paragraphs[0].runs:
                    r.font.color.rgb = WHITE
                    r.font.bold = True

    # --- 4. история: блок 03 «сложности» — реальные инженерные вызовы проекта
    hist_body = [s for s in s_hist.shapes if s.has_text_frame
                 and s.text_frame.text.startswith("Расскажите о самых интересных")][0]
    set_paras(hist_body, [
        "Шум валидации ±0.011 mAP съедал приросты — перешли на парные замеры по 30 "
        "переразбиениям. Скорость упиралась в декодирование JPEG на одном ядре — "
        "пул потоков дал 66 → 126 FPS.",
        "Самая точная модель (ViT-H+) оказалась медленной для стенда — выбрали пару "
        "DINOv3-L + ViT-B и дистиллировали в неё знания ViT-H+.",
    ])

    # --- 5. коротко о решении (шаблон 11)
    s = S[1]
    set_paras(shape(s, "Текст 2"), [
        "Две модели: DINOv3 ViT-L/16 и DINOv2 ViT-B/14 с части-признаками; CosFace + "
        "triplet + память XBM, дистилляция от ViT-H+. Эмбеддинг 3072-d, поиск FAISS. "
        "FastAPI + PostgreSQL + React, Docker офлайн."])
    set_paras(shape(s, "TextBox 27"), [
        f"▪  mAP@10 {n['map10']:.3f} · Rank-1 {n['r1']:.3f} на незнакомых ТС",
        "▪  2 модели, 770 МиБ весов из лимита 2 ГБ",
        f"▪  {n['lat']:.0f} мс на ТС и {n['fps']:.0f} FPS по протоколу стенда",
        "▪  Запуск одной командой: docker compose up, офлайн",
    ])
    set_paras(shape(s, "TextBox 28"), [
        f"▪  Режим отказа: τ = {n['tau']:.3f} по формуле балла 0.7·F1 + 0.3·TNR",
        "▪  Карта внимания: номер модель не использует",
        "▪  Отсев кадров-дублей той же сцены по фону кадра",
        "▪  Готов к 10⁶ галерее: FAISS HNSW 1.1 мс/запрос",
    ])

    # --- путь оператора
    s = S[3]
    set_paras(shape(s, "TextBox 56"), ["Фото или кадр из видео — в браузере или по API. "
                                       "Область автомобиля выделяется рамкой."])
    set_paras(shape(s, "TextBox 58"), [f"Две модели формируют вектор 3072-d за "
                                       f"{n['lat']:.0f} мс. Только внешность — номер "
                                       f"не используется."])
    set_paras(shape(s, "TextBox 64"), ["Карта внимания «почему совпало», экспорт JSON, "
                                       "досье объекта и треки по камерам."])

    # --- как работает ML
    s = S[4]
    set_paras(shape(s, "TextBox 112"), [
        "Два поколения DINO",
        "DINOv3 ViT-L/16 (вес 1.0) + DINOv2 ViT-B/14 (0.7): разные корпуса "
        "предобучения — ошибки декоррелированы.",
        "Части-признаки: 3 полосы кузова со своим эмбеддингом; 1536 + 1536 = 3072-d, "
        "DBA k=2 по галерее.",
    ])
    set_paras(shape(s, "TextBox 113"), [
        "CosFace + triplet + XBM",
        "Угловой отступ (s=48, m=0.25) + batch-hard triplet; память 4096 признаков "
        "(XBM) даёт трудные негативы: +0.008 mAP@10.",
        "Дистилляция точного ViT-H+ (841M) в быструю DINOv3-L; PK-батчи 16×4.",
    ])
    set_paras(shape(s, "TextBox 114"), [
        "Ракурс, свет, перекрытия",
        "Pad + RandomCrop (+0.030 mAP), отражение, color jitter, blur, random "
        "erasing — имитация перекрытий.",
        "Каждый train-ТС снят с 2+ камер: модель учится сближать виды спереди, "
        "сбоку и сзади.",
    ])

    # --- архитектура
    s = S[5]
    set_paras(shape(s, "TextBox 29"), ["кроп +6% → DINOv3-L + DINOv2-B (части-признаки, "
                                       "CUDA-графы) → вектор 3072-d"])
    set_paras(shape(s, "TextBox 31"), ["DBA k=2 по галерее → косинусный поиск FAISS"])
    set_paras(shape(s, "TextBox 35"), ["React + nginx: кадр из видео, кадрирование, "
                                       "порог на лету, экспорт JSON"])
    set_paras(shape(s, "TextBox 37"), ["Python · FastAPI · OpenAPI (Swagger) · "
                                       "JWT и API-ключи · GPU"])
    set_paras(shape(s, "TextBox 38"), ["Хранилища"])
    set_paras(shape(s, "TextBox 39"), ["FAISS-индекс галереи · PostgreSQL: пользователи, "
                                       "история, watchlist"])
    set_paras(shape(s, "TextBox 41"), ["docker compose: postgres + backend + frontend; "
                                       "веса по sha256, офлайн"])

    # --- метрики: нативная диаграмма роста mAP@10 вместо кривой обучения v2
    s = S[6]
    set_paras(shape(s, "TextBox 18"), ["mAP@10"])
    set_paras(shape(s, "TextBox 19"), [f"{n['map10']:.3f}"])
    set_paras(shape(s, "TextBox 20"), ["официальная метрика, 30 переразбиений; было 0.827"])
    set_paras(shape(s, "TextBox 22"), [f"{n['r1']:.3f} / {n['r5']:.3f}"])
    set_paras(shape(s, "TextBox 24"), ["Балл режима отказа"])
    set_paras(shape(s, "TextBox 25"), [f"{n['score']:.3f}"])
    set_paras(shape(s, "TextBox 26"), [f"0.7·F1 + 0.3·TNR при τ = {n['tau']:.3f}"])
    set_paras(shape(s, "TextBox 16"), ["mAP@10 по версиям: одинаковый протокол, 30 сидов"])
    set_paras(shape(s, "TextBox 17"), ["Валидация: 308 незнакомых ТС · 20% запросов без пары "
                                       "· совпадения «та же камера» исключены"])
    old = shape(s, "Picture 15")
    cd = CategoryChartData()
    cd.categories = [c for c, _ in n["map_hist"]]
    cd.add_series("mAP@10", [v for _, v in n["map_hist"]])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, old.left, old.top,
                            old.width, old.height, cd)
    old._element.addprevious(gf._element)
    old._element.getparent().remove(old._element)
    ch = gf.chart
    style_chart(ch, WHITE, 11)
    ch.has_title = False
    ch.has_legend = False
    plot = ch.plots[0]
    plot.gap_width = 60
    ser = plot.series[0]
    for i, (pt_color, (_, v)) in enumerate(zip([ACC2, ACC2, ACC, PINK], n["map_hist"])):
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = pt_color
        # подпись текстом: точка как разделитель, как на остальных слайдах
        dl = pt.data_label
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        dl.text_frame.text = f"{v:.3f}"
        run = dl.text_frame.paragraphs[0].runs[0]
        run.font.bold = True
        run.font.size = Pt(12)
        run.font.color.rgb = WHITE
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = 0.80, 0.90
    va.has_major_gridlines = False
    va.visible = False
    ch.category_axis.tick_labels.font.color.rgb = WHITE
    ch.category_axis.format.line.fill.background()

    # --- режим отказа: кривые F1/TNR/балла по порогу вместо картинок v2
    s = S[7]
    set_paras(shape(s, "TextBox 18"), [f"Порог τ = {n['tau']:.3f}"])
    set_paras(shape(s, "TextBox 19"), ["максимум балла 0.7·F1 + 0.3·TNR (формула "
                                       "постановщика); отбор на сидах 42–61, проверка "
                                       "на 82–111"])
    set_paras(shape(s, "TextBox 20"), [f"F1 = {n['F1']:.3f} · TNR = {n['TNR']:.3f}"])
    set_paras(shape(s, "TextBox 21"), [f"балл {n['score']:.3f}; прежний τ = 0.318 давал "
                                       "TNR 0.44 — сервис отвечал слишком охотно"])
    set_paras(shape(s, "TextBox 23"), ["72% запросов имеют top-1 — кадр-дубль той же "
                                       "камеры. Правило по фону кадра и геометрии BBox "
                                       "отсеивает дубли (precision 0.97–1.0 на train); "
                                       "отказ = нет строк в candidates.csv"])
    pics = [x for x in s.shapes if x.shape_type == 13]
    top_pic, bottom_pic = sorted(pics, key=lambda x: x.top)
    # шаг 0.025 — читаемая ось; сама сетка порогов в threshold.json в 5 раз гуще
    curve = [c for c in n["curve"]
             if 0.25 <= c["tau"] <= 0.70 and round(c["tau"] * 1000) % 25 == 0]
    cd = CategoryChartData()
    cd.categories = [f"{c['tau']:g}" for c in curve]
    cd.add_series("F1", [c["F1"] for c in curve])
    cd.add_series("TNR", [c["TNR"] for c in curve])
    cd.add_series("балл 0.7·F1+0.3·TNR", [0.7 * c["F1"] + 0.3 * c["TNR"] for c in curve])
    gf = s.shapes.add_chart(XL_CHART_TYPE.LINE, top_pic.left, top_pic.top, top_pic.width,
                            bottom_pic.top + bottom_pic.height - top_pic.top, cd)
    top_pic._element.addprevious(gf._element)
    for p in (top_pic, bottom_pic):
        p._element.getparent().remove(p._element)
    ch = gf.chart
    style_chart(ch, DARKTXT, 10)
    ch.has_title = True
    ch.chart_title.text_frame.text = f"Кривые на отложенной валидации · τ = {n['tau']:.3f}"
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    for ser, color, width in zip(ch.plots[0].series, [ACC, ACC2, DARKV], [2.25, 2.25, 3.0]):
        ser.smooth = True
        ser.format.line.color.rgb = color
        ser.format.line.width = Pt(width)
        ser.marker.style = XL_MARKER_STYLE.NONE
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = 0.0, 1.0
    va.major_gridlines.format.line.color.rgb = RGBColor(0xE4, 0xE0, 0xEC)
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(9)
    # подписи через одну: 0.25, 0.30, …

    # --- анализ ошибок: картинки-примеры остаются, выводы — по разбору провалов
    s = S[8]
    set_paras(shape(s, "TextBox 25"), ["Сделано: части-признаки и память XBM. Дальше — "
                                       "агрегация кадров трека."])
    set_paras(shape(s, "TextBox 28"), ["Запрос «в лоб», в галерее — только корма. Верный "
                                       "ответ рядом: у 79% провалов он в топ-10."])
    set_paras(shape(s, "TextBox 29"), ["Дальше: синтетика ракурсов; мультизапрос до 5 "
                                       "снимков уже есть в API."])
    set_paras(shape(s, "TextBox 32"), ["Ночная камера ошибается втрое чаще средней; кропы "
                                       "с пропорцией 2:1 — 89% провалов одной камеры."])
    set_paras(shape(s, "TextBox 33"), ["При низкой уверенности сервис отказывает, а не "
                                       "выдаёт ложную находку."])

    # --- скорость и масштаб
    s = S[9]
    set_paras(shape(s, "TextBox 28"), [f"{n['lat']:.0f} мс"])
    set_paras(shape(s, "TextBox 29"), ["признак одного ТС при батче 1: чтение JPEG → вектор "
                                       "(RTX 4070 Ti SUPER; RTX 3060 — 39 мс)"])
    set_paras(shape(s, "TextBox 30"), [f"{n['fps']:.0f} FPS"])
    set_paras(shape(s, "TextBox 31"), ["пакетно, без роста памяти; пик VRAM 1.9 ГБ "
                                       "(RTX 3060 — 55 FPS)"])
    set_paras(shape(s, "TextBox 32"), ["770 МиБ"])
    set_paras(shape(s, "TextBox 33"), ["2 модели fp16 + индекс + детектор — из лимита "
                                       "2048 МиБ"])
    set_paras(shape(s, "TextBox 35"), ["FAISS HNSW + SQ8 на 10⁶ × 3072-d: 1.1 мс/запрос, "
                                       "recall@10 0.87, индекс 2.9 ГБ"])
    set_paras(shape(s, "TextBox 36"), ["Замеры: src/bench_official.py (протокол стенда, "
                                       "медиана 300 прогонов) и src/ann_demo.py · "
                                       "официальные — скрипт организаторов"])

    # --- демо: скриншоты React-клиента
    s = S[10]
    pics = sorted([x for x in s.shapes if x.shape_type == 13], key=lambda x: x.left)
    replace_picture(s, pics[0], Path(shots) / "ui_results.png", width=Inches(4.62))
    replace_picture(s, pics[1], Path(shots) / "ui_analytics.png", width=Inches(4.62))
    for grp in [x for x in s.shapes if x.shape_type == 6]:
        for sub in grp.shapes:
            if sub.has_text_frame and "localhost" in sub.text_frame.text:
                sub.text_frame.paragraphs[0].runs[0].text = "localhost:8080"
                for r in sub.text_frame.paragraphs[0].runs[1:]:
                    r.text = ""
    set_paras(shape(s, "TextBox 58"), ["Поиск: кандидаты, уверенность, карта внимания"])
    set_paras(shape(s, "TextBox 59"), ["Аналитика: камеры, межкамерные треки, история"])
    set_paras(shape(s, "TextBox 60"), ["▪  Фото или кадр из видео, кадрирование в браузере"])
    set_paras(shape(s, "TextBox 61"), ["▪  Порог отказа и top-N меняются на лету"])
    set_paras(shape(s, "TextBox 62"), ["▪  Карта внимания: куда смотрела модель"])
    set_paras(shape(s, "TextBox 63"), ["▪  Экспорт результата в JSON · Swagger на /docs"])

    # --- соответствие ТЗ
    s = S[11]
    set_paras(shape(s, "TextBox 23"), ["submission.csv · embeddings.npy · candidates.csv — "
                                       "самопроверка пройдена, в Docker воспроизводятся "
                                       "(cos ≥ 0.99999)"])
    set_paras(shape(s, "TextBox 25"), ["docker compose up → UI :8080 + API :8000 · "
                                       "docker compose run --rm inference → артефакты; офлайн"])
    set_paras(shape(s, "TextBox 27"), ["OpenAPI/Swagger на /docs · React-клиент в браузере "
                                       "· JWT и API-ключи"])
    set_paras(shape(s, "TextBox 29"), ["веса 770 МиБ ≤ 2048 · без OOM (пик VRAM 1.9 ГБ) · "
                                       "точные версии пакетов"])
    set_paras(shape(s, "TextBox 31"), ["признаки номера не используются: номера размыты, "
                                       "карта внимания на номер не смотрит"])

    # --- развитие
    s = S[12]
    set_paras(shape(s, "TextBox 12"), ["Агрегация кадров трека · ViT-H+ как «точный профиль» "
                                       "(mAP@10 0.883) для мощных GPU · калибровка порога "
                                       "на живом потоке"])

    # примеры из тестовой выборки — сразу после анализа ошибок
    add_test_errors_slide(prs, ROOT / "artifacts/figures/test_examples_v4.png")

    # порядок: обязательный блок шаблона 7–11, затем презентация решения
    new_team, new_hist, new_test = len(S), len(S) + 1, len(S) + 2
    reorder(prs, [0, 13, new_team, new_hist, 1, 2, 3, 4, 5, 6, 7, 8, new_test,
                  9, 10, 11, 12])
    prs.save(out)
    print("saved:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--shots", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.deck, a.template, a.shots, a.out)
