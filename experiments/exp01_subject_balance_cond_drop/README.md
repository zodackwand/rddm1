# Exp 01: балансировка по людям и dropout условия

## Зачем

`rddm_v1` и `rddm_v2` запоминают обучающих людей. На train-людях ошибка пульса 1.3–1.8 уд/мин, на test-людях WESAD и DALIA 7–15.
Хуже всего там, где людей мало, а записи длинные: в WESAD и DALIA по 12 человек в train, по 1 400–2 100 окон на человека. На клинических датасетах (по 120–300 окон на человека) провала нет.

Проверяем два регуляризатора:
- **Балансировка по людям** (`--per-subject 300`): каждую эпоху берётся не больше 300 случайных окон от человека. Окна меняются от эпохи к эпохе.
- **Dropout условия** (`--cond-drop 0.1`): у 10% примеров в батче обнуляются признаки PPG после `ConditionNet`. В конфиге авторов `cond_mask = 0.0`, так что это наша идея, а не повтор их рецепта.

## Запуски

Все запуски: 1 GPU, батч 128, 500 эпох, cosine lr, остальное как у v1. Сравниваем с `rddm_v1`.

| Запуск | Флаги | Окон за эпоху | Чекпойнты |
|---|---|---|---|
| `bal300` | `--per-subject 300` | ~24.7 тыс. | `runs/exp01_bal300` |
| `bal300_cd01` | `--per-subject 300 --cond-drop 0.1` | ~24.7 тыс. | `runs/exp01_bal300_cd01` |
| `cd01` | `--cond-drop 0.1` | 60.5 тыс. | `runs/exp01_cd01` |

С балансировкой шагов за 500 эпох примерно в 2.4 раза меньше, чем у v1.
Ожидаемое время: около суток для `bal300*`, около двух для `cd01`. Чекпойнт каждые 50 эпох.

## Запуск на Avon

Из корня репозитория:

```bash
git pull
bash experiments/exp01_subject_balance_cond_drop/submit.sh
squeue --me
```

`submit.sh` ставит в очередь 10 задач с зависимостями:
1. `preprocess.py --ids-only` (CPU, до 4 ч). Пишет `datasets/<name>/subj_<part>_<sec>sec.npy`, номер человека для каждого окна, и сверяет число окон и разбиение с уже готовыми файлами.
2. Три обучения, стартуют после шага 1.
3. После каждого обучения: `evaluate.py` (RMSE, FD, пульс на всём тесте) и `diag2_epoch_curve.py` (пульс по чекпойнтам на 600 окнах WESAD и DALIA, train и test).

Логи: `experiments/exp01_subject_balance_cond_drop/logs/`. Результаты: `experiments/exp01_subject_balance_cond_drop/results/<запуск>/`.

### Если обучение упёрлось в лимит 48 ч

Оценка для этого запуска не стартует (зависимость не выполнится). Нужно убрать её и продолжить с `last.pt` тем же путём:

```bash
squeue --me   # задачи exp01-eval-* и exp01-curve-* в состоянии DependencyNeverSatisfied
scancel <id eval> <id curve>
sbatch --output=experiments/exp01_subject_balance_cond_drop/logs/train_cd01_%j.log \
  train.slurm --path runs/exp01_cd01 --cond-drop 0.1 --save-every 50
```

Когда допишется эпоха 499, оценку запустить руками:

```bash
EXP=experiments/exp01_subject_balance_cond_drop
sbatch --output=$EXP/logs/eval_cd01_%j.log evaluate.slurm --path runs/exp01_cd01 --out $EXP/results/cd01/eval
sbatch --time=08:00:00 --output=$EXP/logs/curve_cd01_%j.log diagnostics.slurm diag2_epoch_curve.py \
  --path runs/exp01_cd01 --out $EXP/results/cd01/curve.json
```

### Проверка по ходу

```bash
tail -n 3 experiments/exp01_subject_balance_cond_drop/logs/train_*.log
grep -h "ids for\|Error" experiments/exp01_subject_balance_cond_drop/logs/ids_*.log
```

В начале лога `bal300*` должна быть строка `subjects: 128, windows per epoch: ...`.

## Что забрать на ноутбук

Результаты (`metrics.json`, `curve.json`, картинки и `.npz` с генерацией, около 110 МБ) и логи. Чекпойнты не нужны.
Команды запускать **на ноутбуке**, `AVON` — твой логин на кластере:

```bash
export AVON=<логин>@avon.scrtp.warwick.ac.uk
REMOTE='~/research/theoretical_signals/rddm1/experiments/exp01_subject_balance_cond_drop'
LOCAL=~/research/theoretical_signals/local_results/exp01
mkdir -p $LOCAL
rsync -avz "$AVON:$REMOTE/results/" $LOCAL/results/
rsync -avz "$AVON:$REMOTE/logs/" $LOCAL/logs/
```

Если нужны только цифры, без `.npz`:

```bash
rsync -avz --include='*/' --include='*.json' --include='*.png' --include='*.log' --exclude='*' \
  "$AVON:$REMOTE/" $LOCAL/
```

## Как читать результат

Главная метрика: MAE пульса на **test** WESAD и DALIA (`results/<запуск>/eval/metrics.json`, поле `mae_hr`). У v1: 9.31 и 9.76, у весов авторов 1.99 и 2.57, в статье 1.40 и 4.49.
- **Test-MAE заметно падает (до 5 и ниже), а train-MAE в `curve.json` растёт:** регуляризатор работает, разрыв train/test сокращается.
- **Test-MAE остаётся 8–10:** запоминание людей этими средствами не лечится, искать дальше (аугментации, меньшая модель, ранняя остановка).
- RMSE и FD не должны заметно ухудшиться относительно v1.
