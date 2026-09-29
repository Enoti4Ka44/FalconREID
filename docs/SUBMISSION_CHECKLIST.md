# Чек-лист сдачи (п. 8 и 13 ТЗ)

## Артефакты в репозитории

| Артефакт | Файл | Статус |
|---|---|---|
| submission.csv (топ-10 на запрос) | `backend/artifacts/submission.csv` | ✔ самопроверка пройдена |
| embeddings.npy (query, затем gallery, порядок CSV) | `backend/artifacts/embeddings.npy` | ✔ 1860 × 3072, float32, L2 |
| candidates.csv (принятые кандидаты; отказ = нет строк) | `backend/artifacts/candidates.csv` | ✔ 100 отказов из 1110 |
| Код обучения и инференса | `backend/src/` | ✔ |
| Dockerfile / docker-compose.yml | `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` | ✔ проверено: сборка, стек, `run --rm inference` |
| Файлы зависимостей (точные версии) | `backend/requirements.txt`, `backend/requirements.docker.txt`, `frontend/package-lock.json` | ✔ |
| Веса в составе решения | релиз `weights-v4` + `backend/models/weights.json` (sha256), скачиваются при сборке | ✔ загружены |
| README с внешними ресурсами | `README.md` | ✔ |
| Сопроводительная документация | `docs/SOLUTION.md`, `docs/ФальконТех_документация.pdf` | ✔ |
| Презентация | `docs/bryansk1.pptx` / `docs/bryansk1.pdf` | ✔ 11 слайдов; обязательные слайды шаблона 7–11 сохранены |

Самопроверка перед сдачей:

```bash
cd backend && python -m src.validate_submission --data-root <DATA_DIR> --artifacts artifacts
```

## Ссылки для платформы (п. 13)

1. **Репозиторий** — https://github.com/Enoti4Ka44/FalconREID (**сделать публичным**:
   Settings → General → Danger Zone → Change visibility → Public; без этого жюри
   не увидит ни код, ни релиз с весами).
2. **Презентация** — `docs/bryansk1.pdf` в репозитории
   (или копия на диске с доступом «все, у кого есть ссылка»).
3. **Прототип** — https://reid.up-point.tech/app (логин `test` / `test`), Swagger — https://reid.up-point.tech/docs;
   видео работы интерфейса — `docs/falcon_demo.mp4`.
4. **Документация** — `docs/ФальконТех_документация.pdf` (или `README.md`).

## Ручные действия перед сдачей

- [x] Вписать название команды (титул) и данные участников — команда «Брянск‑1», 5 участников (слайды 1–3)
- [x] Сделать репозиторий публичным и проверить из окна инкогнито: README, релиз `weights-v4`
- [x] Прототип развёрнут: https://reid.up-point.tech/app (логин `test` / `test`), видео — `docs/falcon_demo.mp4`
- [ ] Вставить ссылки на платформу до 29 сентября 23:59 МСК; после стоп-кода не коммитить
