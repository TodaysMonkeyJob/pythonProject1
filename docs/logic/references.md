<!-- Згенеровано: python -m tools.logic_docs (tools/logic_docs/text/*.md + config). Не редагуйте вручну — змініть шаблон і запустіть знову. -->

# Джерела, покажчик формул і словник

## Джерела

[1] NIOSH. Immediately Dangerous to Life or Health (IDLH) Values. Centers for Disease Control and Prevention,
National Institute for Occupational Safety and Health. https://www.cdc.gov/niosh/idlh/ — критичні рівні газів
(перевірено 06–08.11.2026).

[2] NIOSH. Pocket Guide to Chemical Hazards. https://www.cdc.gov/niosh/npg/ — IDLH, нижні межі вибуховості (LEL),
молярні маси; сторінки ацетону (npgd0004), аміаку (npgd0028), CO (npgd0105), етанолу (npgd0262), сірководню
(npgd0337), толуолу (npgd0619); перевірено 08.11.2026.

[3] U.S. EPA. Final Updates to the Air Quality Index (AQI) for Particulate Matter, 2024 (чинні з 06.05.2024):
категорія «Hazardous» для PM2.5 (24 год) починається з 225,5 мкг/м³. https://www.epa.gov/

[4] OSHA. 29 CFR 1910.1000(d)(2)(i): формула для сумішей речовин зі схожою дією (сума cᵢ / Lᵢ не має перевищувати
одиницю). https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.1000

[5] Zadeh L. A. Fuzzy sets // Information and Control. 1965. Vol. 8, No. 3. P. 338–353.

[6] Mamdani E. H., Assilian S. An experiment in linguistic synthesis with a fuzzy logic controller // International
Journal of Man-Machine Studies. 1975. Vol. 7, No. 1. P. 1–13.

[7] Mendel J. M. Uncertain Rule-Based Fuzzy Systems: Introduction and New Directions. 2nd ed. Springer, 2017.

[8] Karnik N. N., Mendel J. M. Centroid of a type-2 fuzzy set // Information Sciences. 2001. Vol. 132, No. 1–4.
P. 195–220.

[9] Ruspini E. H. A new approach to clustering // Information and Control. 1969. Vol. 15, No. 1. P. 22–32.

[10] Wilson E. B. Probable inference, the law of succession, and statistical inference // Journal of the American
Statistical Association. 1927. Vol. 22, No. 158. P. 209–212.

[11] ANSI S3.5-1997. American National Standard Methods for Calculation of the Speech Intelligibility Index —
рівні мовлення: звичайний, підвищений, голосний, крик.

[12] Zivkovic Z. Improved adaptive Gaussian mixture model for background subtraction // Proceedings of the 17th
International Conference on Pattern Recognition (ICPR). 2004. Vol. 2. P. 28–31 — модель фону, на якій побудовано
детектор руху.

[13] Паспорти й описи сенсорів (точні посилання на сторінки потрібно підтвердити): DFRobot Wiki, SKU SEN0469 (NH₃) і
SEN0466 (CO) — https://wiki.dfrobot.com; Nova Fitness SDS011 (паспорт); Sensirion SPS30 (паспорт) —
https://sensirion.com; Winsen MQ-137, MQ-7 — https://www.winsen-sensor.com; ScioSense ENS160 (паспорт) —
https://www.sciosense.com. Значення, які взято лише з вторинних джерел, позначено в таблиці сенсорів.

[14] Власні публікації автора про шумозаглушення, аудіосистему та електронний ніс — **бібліографічні дані потрібно
додати** (у репозиторії їх немає; документ навмисно їх не вигадує). Графеновий сенсор і його модель описано за
нашими лабораторними даними.

[15] Desplanques B., Thienpondt J., Demuynck K. ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation
in TDNN Based Speaker Verification // Proceedings of Interspeech 2020. P. 3830–3834 — впізнавання мовця.

[16] Radford A., Kim J. W., Xu T., Brockman G., McLeavey C., Sutskever I. Robust Speech Recognition via Large-Scale
Weak Supervision // Proceedings of the 40th International Conference on Machine Learning (ICML). 2023 — Whisper.

[17] Huang G. B., Ramesh M., Berg T., Learned-Miller E. Labeled Faces in the Wild: A Database for Studying Face
Recognition in Unconstrained Environments. University of Massachusetts, Amherst, Technical Report 07-49, 2007;
Zhong Y., Deng W., Hu J., Zhao D., Li X., Wen D. SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face
Recognition // IEEE Transactions on Image Processing. 2021. Vol. 30. P. 2587–2598 — впізнавання облич і його
перевірка.

[18] Jøsang A., Ismail R. The Beta Reputation System // Proceedings of the 15th Bled Electronic Commerce
Conference. 2002 — модель довіри з підтвердженнями й спростуваннями (не використовується в статті 1).

[19] Kuehne H., Jhuang H., Garrote E., Poggio T., Serre T. HMDB: A Large Video Database for Human Motion Recognition
// ICCV 2011; Shahroudy A., Liu J., Ng T.-T., Wang G. NTU RGB+D: A Large Scale Dataset for 3D Human Activity
Analysis // CVPR 2016 — дані, на яких виміряно точність розпізнавання дій.

[20] Wooldridge M. An Introduction to MultiAgent Systems. 2nd ed. Wiley, 2009.

[21] Jocher G., Qiu J. Ultralytics YOLO11. 2024. https://github.com/ultralytics/ultralytics — виявлення людей і
предметів у кадрі.

## Покажчик формул

Кожна формула документа: де вона використовується і звідки взята. «Загальновідома» — формула з будь-якого підручника
(трикутна функція, центр мас, переведення відсотків).

*Таблиця 18.*

| № | Формула | Де використовується | Джерело |
| -- | -- | -- | -- |
| Ф1 | сила правила = добуток належностей його умов | усі бази правил | [6], [7] |
| Ф2 | ризик = Σ wᵣ · cᵣ / Σ wᵣ (центр мас; wᵣ — сила правила, cᵣ — центр його вихідного слова) | усі бази правил | загальновідома [6] |
| Ф3 | 1 об.% = 10 000 ppm | газ: переведення одиниць | загальновідома |
| Ф4 | мг/м³ = ppm · M / 24,45 (25 °C, 1 атм) | газ: переведення одиниць | [2] |
| Ф5 | частка критичного рівня = c / C_крит | газ: порівняння всіх речовин | [1], [3] (значення C_крит) |
| Ф6 | трикутна функція належності: μ(x) = max(0, min((x − a)/(b − a), (c − x)/(c − b))) | усі входи й виходи | загальновідома [5] |
| Ф7 | трапецієвидна: μ(x) = max(0, min((x − a)/(b − a), 1, (d − x)/(d − c))) | усі входи й виходи | загальновідома [5] |
| Ф8 | швидкість = нахил прямої найменших квадратів за останні 60 с, × 60 (за хвилину) | газ: динаміка | загальновідома |
| Ф9 | індекс суміші = Σ cᵢ / C_крит,ᵢ ≥ 1 → критично (лише в одній групі дії) | газ: суміш газів | [4] |
| Ф10 | частка незнайомих = N_незнайомих / (N_незнайомих + N_відомих) | камера | вимога керівника (слайди 9–11) |
| Ф11 | гучність над фоном = рівень кадру − фон (20-й перцентиль медіан за 5 хв), дБ | мікрофон | загальновідома (перцентиль) |
| Ф12 | загальний рівень = найвищий + 1, якщо друга підсистема активна; критичний — лише з доказом | верхній рівень | вимога керівника (TOD-51), правило системи |
| Ф13 | смуга (тип-2): верхня крива = ребра зсунуті назовні на δ, нижня — всередину на δ | коридор | [7] |
| Ф14 | коридор [y_l, y_r] — алгоритм Карніка–Менделя (центр множин) | коридор | [8] |
| Ф15 | нижня межа Вілсона: (p̂ + z²/2n − z·√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n), z = 1,96 | вага агентів, точність каналів | [10] |

## Словник

- **Агент** — окрема програма з однією роллю, яка робить заяви, але не ухвалює рішення сама (частина 2).
- **Валідаційний агент** — агент, який зважує заяви інших і ухвалює остаточне рішення в межах коридору.
- **Група дії** — гази зі схожою дією на організм; їхні частки критичного рівня складаються.
- **Заява** — повідомлення агента: що він побачив, з якою впевненістю, на яких подіях ґрунтується.
- **Калібрування** — вимірювання власних характеристик конкретного сенсора; зберігається і перевіряється під час
  запуску.
- **Коридор [y_l, y_r]** — проміжок, у якому лежить ризик з урахуванням похибок сенсорів.
- **Критичний рівень (C_крит)** — концентрація, з якої починається безпосередня небезпека для життя (IDLH).
- **Монотонність** — властивість: гірший вхід ніколи не зменшує ризик.
- **НКМПР (LEL)** — нижня концентраційна межа поширення полум'я: з неї суміш може вибухнути.
- **Нечітка логіка** — логіка, де висловлення може бути істинним частково, від 0 до 1.
- **Нечітка множина типу 2** — множина, у якої належність — проміжок, а не одне число; так враховують похибку.
- **ppm** — мільйонна частка: одна молекула газу на мільйон молекул повітря.
- **Правило** — «якщо … то …» зі словами (термами) замість чисел.
- **Розбиття Руспіні** — набір кривих, які в кожній точці в сумі дають 1.
- **Терм (слово)** — словесне значення входу («безпечно», «помірно»), описане функцією належності.
- **Функція належності** — крива, яка показує, наскільки значення підходить під слово: від 0 до 1.
- **EER** — частка помилок у точці, де хибні прийняття дорівнюють хибним відмовам.
- **IDLH** — Immediately Dangerous to Life or Health, критичний рівень за NIOSH.
- **Macro-F1** — середня якість розпізнавання за всіма рівнями ризику: 1 — ідеально, 0 — жодного влучення.
