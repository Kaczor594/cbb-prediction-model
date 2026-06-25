library(tidyverse)
library(plotly)

# Load SHAP values for travel features
shap <- read_csv("data/travel_shap_values.csv", show_col_types = FALSE)

# Reshape to long format: one row per (game, feature)
long <- bind_rows(
  shap %>%
    transmute(
      distance = home_travel_distance,
      shap_contribution = home_travel_shap,
      feature = "home_travel_distance",
      neutral_site = factor(neutral_site, levels = c(0, 1), labels = c("Home/Away", "Neutral")),
      home_winner = home_winner
    ),
  shap %>%
    transmute(
      distance = away_travel_distance,
      shap_contribution = away_travel_shap,
      feature = "away_travel_distance",
      neutral_site = factor(neutral_site, levels = c(0, 1), labels = c("Home/Away", "Neutral")),
      home_winner = home_winner
    )
)

# Arkansas vs High Point reference points
ark_hp <- tibble(
  distance = c(1617.27, 2299.38),
  shap_contribution = c(-0.4883, -0.3546),
  feature = c("home_travel_distance", "away_travel_distance"),
  label = c("Arkansas (1,617 mi)", "High Point (2,299 mi)")
)

# For smoothing, only use points with distance > 0 (home_travel is 0 for 88% of games)
long_nonzero <- long %>% filter(distance > 10)

# Compute binned means for a cleaner trend line
long_binned <- long_nonzero %>%
  mutate(dist_bin = cut(distance, breaks = seq(0, 5200, by = 200), include.lowest = TRUE)) %>%
  group_by(feature, dist_bin) %>%
  summarise(
    distance = mean(distance),
    shap_contribution = mean(shap_contribution),
    n = n(),
    .groups = "drop"
  ) %>%
  filter(n >= 5)  # only show bins with enough data

# Also compute the value at distance = 0 for home_travel_distance
home_at_zero <- long %>%
  filter(feature == "home_travel_distance", distance == 0) %>%
  summarise(
    distance = 0,
    shap_contribution = mean(shap_contribution),
    n = n(),
    feature = "home_travel_distance"
  )

away_at_zero <- long %>%
  filter(feature == "away_travel_distance", distance < 10) %>%
  summarise(
    distance = 0,
    shap_contribution = mean(shap_contribution),
    n = n(),
    feature = "away_travel_distance"
  )

zero_pts <- bind_rows(home_at_zero, away_at_zero)

# ── Static ggplot ──

p <- ggplot() +
  # All individual points
  geom_point(data = long, aes(x = distance, y = shap_contribution, color = feature),
             alpha = 0.08, size = 0.5) +
  # Binned trend line
  geom_line(data = long_binned, aes(x = distance, y = shap_contribution, color = feature),
            linewidth = 1.3) +
  geom_point(data = long_binned, aes(x = distance, y = shap_contribution, color = feature),
             size = 1.5, alpha = 0.7) +
  # Zero-distance reference points
  geom_point(data = zero_pts, aes(x = distance, y = shap_contribution, color = feature),
             size = 3, shape = 15) +
  # Reference line
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  # Arkansas vs High Point annotations
  geom_point(data = ark_hp, aes(x = distance, y = shap_contribution),
             size = 4, shape = 18, color = "black") +
  geom_label(data = ark_hp, aes(x = distance, y = shap_contribution, label = label),
             color = "black", size = 3, nudge_y = 0.06, fill = "white", label.size = 0.3) +
  scale_color_manual(
    values = c("home_travel_distance" = "#E63946", "away_travel_distance" = "#457B9D"),
    labels = c("Home Team Travel", "Away Team Travel")
  ) +
  labs(
    title = "SHAP Contribution of Travel Distance Features",
    subtitle = "How travel distance affects XGBoost's prediction (positive = favors home team)\nLine shows binned average (200-mile bins, min 5 games)",
    x = "Travel Distance (miles)",
    y = "SHAP Contribution (log-odds)",
    color = "Feature"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "top",
    plot.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

print(p)

# ── Interactive plotly version ──

p_plotly <- ggplot() +
  geom_point(data = long, aes(x = distance, y = shap_contribution, color = feature,
    text = paste0(
      "Distance: ", round(distance, 0), " mi\n",
      "SHAP: ", sprintf("%+.4f", shap_contribution), "\n",
      "Site: ", neutral_site
    )), alpha = 0.08, size = 0.5) +
  geom_line(data = long_binned, aes(x = distance, y = shap_contribution, color = feature),
            linewidth = 1.3) +
  geom_point(data = long_binned, aes(x = distance, y = shap_contribution, color = feature,
    text = paste0(
      "Bin avg distance: ", round(distance, 0), " mi\n",
      "Avg SHAP: ", sprintf("%+.4f", shap_contribution), "\n",
      "Games in bin: ", n
    )), size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  geom_point(data = ark_hp, aes(x = distance, y = shap_contribution),
             size = 4, shape = 18, color = "black") +
  scale_color_manual(
    values = c("home_travel_distance" = "#E63946", "away_travel_distance" = "#457B9D"),
    labels = c("Home Team Travel", "Away Team Travel")
  ) +
  labs(
    title = "SHAP Contribution of Travel Distance Features",
    x = "Travel Distance (miles)",
    y = "SHAP Contribution (log-odds)",
    color = "Feature"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "top",
    plot.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

fig <- ggplotly(p_plotly, tooltip = "text") %>%
  layout(
    title = list(
      text = "SHAP Contribution of Travel Distance Features<br><sup>Positive = favors home team | Line = 200-mile binned average</sup>"
    ),
    annotations = list(
      list(x = 1617, y = -0.4883, text = "Arkansas<br>(home)", showarrow = TRUE,
           arrowhead = 2, ax = -50, ay = -35, font = list(size = 11, color = "#E63946")),
      list(x = 2299, y = -0.3546, text = "High Point<br>(away)", showarrow = TRUE,
           arrowhead = 2, ax = 50, ay = -35, font = list(size = 11, color = "#457B9D"))
    )
  )

fig
