Use this prompt to create a report-ready image that explains the full code and data flow of the project.
text
Create a publication-quality research pipeline diagram for a Korean environmental AI project.

Title:
"Daecheong Dam Area-Specific Harmful Cyanobacteria Early Warning and Decision Support Pipeline"

Style:
Clean academic infographic, white background, vector-like layout, restrained blue, teal, green and gray palette with small orange warning accents. No emojis, no cartoon characters, no decorative blobs. Use thin arrows, compact labels, and a professional environmental data science style suitable for a technical report.

Canvas:
Wide 16:9 horizontal layout. Left-to-right flow. Korean main labels with short English subtitles.

Main pipeline blocks:

1. Data Collection / 데이터 수집
   - Selenium crawler for dam operation data: water level, storage, storage rate, inflow, outflow, dam rainfall.
   - Selenium crawler for water quality and algae data: water temperature, pH, DO, turbidity, transparency, Chl-a, harmful cyanobacteria cells, Microcystis, Anabaena, Oscillatoria, Aphanizomenon.
   - Weather API: temperature, rainfall, wind, sunshine, solar radiation, cloud cover.
   - Show three Daecheong Dam areas: Munui, Chudong, Hoenam.

2. Raw Data Store / 원천 데이터 저장
   - data/대청수문_10년치_통합데이터.csv
   - data/수질_10년치_통합데이터.csv
   - data/WEATHER.csv
   - water_data/dam and water_data/quality Excel downloads.

3. Preprocessing / 전처리
   - Align by date and sampling location.
   - Merge dam operation, water quality, algae, and weather tables.
   - Validate duplicate date plus location keys, missing values, and numeric feature columns.
   - Output: data/Final.csv and data/processed/model_input/algae_model_input.csv.

4. Feature Engineering / 파생변수 생성
   - Rolling weather windows: rain_3d_sum, rain_7d_sum, rain_14d_sum, air_temp_7d_mean, wind_7d_mean.
   - Hydrology features: inflow_7d_sum, outflow_7d_sum, residence_proxy, nutrient_stagnation_index.
   - Temporal features: sampling_gap_days, previous_observed_cells, previous_exceeded.
   - Spatial/upstream features: loc_flow_order, upstream_cells_same_date, hoenam_pressure_for_downstream.
   - Include formula box: next_log_cells = log10(next cells + 1).

5. Model Training / 모델 학습
   - YAML-configured candidate models.
   - Regression head: predict next_log_cells.
   - Classification head: predict next_alert_binary.
   - Show model boxes: LightGBM, XGBoost, Random Forest, Huber Regressor, HistGradientBoosting, CatBoost, Stacking Ensemble.
   - Time-based validation split, not random split.

6. Evaluation / 모델 평가
   - Regression metrics: RMSE, MAE, RMSE_cells, MAE_cells, RMSLE.
   - Classification metrics: recall, precision, F1, ROC AUC, PR AUC.
   - Compare against persistence baseline.
   - Emphasize recall because missing a bloom warning is high-risk.

7. Explainability / 설명 가능성
   - Feature importance and SHAP top reasons.
   - Example factors: high water temperature, long residence time, low outflow, rainfall and inflow, low wind, high sunshine, upstream Hoenam cell increase.

8. Scenario Decision Layer / 의사결정 시나리오
   - Alert thresholds: watch >= 1,000 cells/mL, warning >= 10,000 cells/mL, bloom >= 1,000,000 cells/mL.
   - Rule box: previous_exceeded AND predicted_exceeded = operational alert candidate.
   - Scenario types: watch candidate, warning candidate, bloom monitoring, rainfall nutrient inflow, stagnant stratification risk, high temperature and sunlight growth, Hoenam upstream propagation, general stable.

9. Outputs / 산출물
   - Predicted cells.
   - Alert probability.
   - Predicted alert stage.
   - Top SHAP reasons.
   - Scenario results.
   - SVG model comparison chart.
   - Saved models and metrics under artifacts/.

Layout details:
Place "Data Collection" on the far left as three stacked source cards. Put "Preprocessing" and "Feature Engineering" in the center as connected processing blocks. Split into two parallel model heads, Regression and Classification, then merge into the Scenario Decision Layer. Put final report/dashboard outputs on the far right. Add a small map-like strip for Munui, Chudong, Hoenam near the data source area. Keep all text legible and avoid clutter.