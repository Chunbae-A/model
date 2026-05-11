# Pipeline Image Prompt

Create a publication-quality research pipeline diagram for a Korean water-quality AI project.

Topic: "Daecheong Dam Area-Specific Harmful Cyanobacteria Early Warning and Decision Support System".

Style:
- Academic paper figure, clean vector-like infographic.
- White background, restrained blue/teal/green palette with small orange warning accents.
- Horizontal left-to-right pipeline with clear numbered blocks.
- Use thin arrows, grouped modules, small icons, and compact labels.
- Make it look like a figure from an environmental machine learning paper, not a marketing slide.
- Use Korean labels with small English subtitles.

Required pipeline blocks:
1. Data Sources / 데이터 수집
   - Weekly water-quality and algae sampling data: water temperature, pH, DO, turbidity, transparency, Chl-a, harmful cyanobacteria cell count, Microcystis, Anabaena, Oscillatoria, Aphanizomenon.
   - Dam operation and hydrology data: water level, storage, storage rate, inflow, outflow.
   - Weather data: air temperature, rainfall, wind speed, sunshine duration, solar radiation, cloud cover.
   - Sampling locations: Munui, Hoenam, Chudong.

2. Time Alignment and Feature Engineering / 시간 정렬 및 피처 생성
   - Align by sampling date and location.
   - Aggregate weather and hydrology over 3-day, 7-day, and 14-day windows.
   - Create sampling_gap_days for irregular sampling intervals.
   - Create hydrological features: residence_proxy, inflow_7d_sum, outflow_7d_sum, nutrient_stagnation_index.
   - Create weather features: air_temp_7d_mean, rain_3d_sum, rain_7d_sum, rain_14d_sum, wind_7d_mean, sunshine_7d_sum, solar_7d_sum.
   - Create spatial/upstream features: loc_flow_order, Hoenam pressure, upstream same-date cell counts.

3. Target Definition / 타깃 정의
   - Main target: harmful cyanobacteria total cell count.
   - Regression target: next_log_cells = log10(next harmful cyanobacteria cells + 1).
   - Classification target: next_alert_binary = whether next cells exceed 1,000 cells/mL.
   - Operational logic: previous_exceeded AND predicted_exceeded for two-consecutive-sampling alert candidate.

4. Model Training / 모델 학습
   - YAML-configured candidate models.
   - Show six model boxes: LightGBM, XGBoost, RandomForest, HistGradientBoosting, CatBoost, Stacking Ensemble.
   - Stacking Ensemble should receive arrows from the five base models and output a combined prediction.
   - Use time-based validation, not random split.
   - Show regression and classification as two parallel heads.

5. Evaluation / 성능 평가
   - Regression metrics: RMSE, MAE, RMSE_cells, MAE_cells, RMSLE.
   - Classification metrics: Recall, Precision, F1, ROC AUC, PR AUC.
   - Emphasize Recall because missing a bloom warning is the most dangerous error.
   - Compare against Persistence Baseline.

6. Explainability / 원인 해석
   - SHAP top factors.
   - Example factors: water temperature increase, residence time increase, low outflow, rainfall and inflow, low wind, high sunshine, Hoenam cell increase.

7. Scenario Decision Layer / 시나리오 의사결정
   - Administrative alert logic: watch >= 1,000, warning >= 10,000, bloom >= 1,000,000 cells/mL.
   - Scenario categories: watch alert candidate, warning alert candidate, bloom monitoring, approaching watch threshold, downgrade/release observation, stagnant stratification risk, rainfall nutrient inflow, high temperature and sunlight growth, Hoenam upstream propagation, general stable.
   - Link scenarios to recommended monitoring/action categories.

8. Final Output / 최종 산출물
   - 7-day ahead harmful cyanobacteria cell prediction.
   - Alert probability.
   - Predicted alert stage.
   - Top SHAP reasons.
   - LLM-ready scenario briefing payload.
   - SVG model comparison chart.

Layout details:
- Put "Data Sources" on the far left as three stacked cylinders/cards.
- Put "Feature Engineering" as a central processing block with small sub-feature chips.
- Split into two model heads: Regression and Classification.
- Merge outputs into "Legal Alert Logic + Scenario Engine".
- Final rightmost panel: dashboard/report outputs for water managers.
- Add a small map-like strip showing the three Daecheong Dam areas: Munui, Hoenam, Chudong.
- Include a small formula box: log10(cells + 1).
- Include a small rule box: previous exceeded + predicted exceeded = alert candidate.
- Avoid decorative blobs, 3D, or cartoon style.
- Keep text legible and professional.
