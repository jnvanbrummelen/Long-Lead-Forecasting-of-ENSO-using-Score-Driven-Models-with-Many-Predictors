# ==============================================================================
# This script helps visualize ENSO cyclicity and test for nonstationary seasonality; 
# it is used to create Figure 1 and Tables A.1 and A.2.
#
# Dependencies:
#   readxl
#   here
#   lubridate
#   uroot
#   sysfonts
#   showtext
#
# By Sebastian Jensen
# Aarhus University
# CoRE
# Jul, 2026
# ==============================================================================

# -----------------------------------------------------------------------------
# R environment
# -----------------------------------------------------------------------------
rm(list=ls())

sysfonts::font_add_google("Lora", "lora")
showtext::showtext_auto()

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
data_file <- here::here("../Data", "Book1982_2025_autosim_HIGHRES_OISST.xlsx")
ENSO_data <- readxl::read_excel(data_file)

dates <- ENSO_data$DATE
dates <- lubridate::ym(dates)

EN34 <- ENSO_data$EN3.4

# -----------------------------------------------------------------------------
# Sample splits
# -----------------------------------------------------------------------------
# In-sample period
in_sample_start_year <- 1982L
in_sample_start_month <- 1L

in_sample_end_year <- 2004L
in_sample_endt_month <- 12L

in_sample_begin <- lubridate::make_date(in_sample_start_year, in_sample_start_month, 1)
in_sample_end <- lubridate::make_date(in_sample_end_year, in_sample_endt_month, 1)

idx_begin <- match(in_sample_begin, dates)
idx_end <- match(in_sample_end, dates)

EN34_in <- EN34[idx_begin:idx_end]
dates_in <- dates[idx_begin:idx_end]

x_in <- ts(EN34_in, start = c(in_sample_start_year, in_sample_start_month), frequency = 12)

# Out-of-sample period
out_sample_start_year <- 2005L
out_sample_start_month <- 1L

out_sample_end_year <- 2024L
out_sample_endt_month <- 12L

out_sample_begin <- lubridate::make_date(out_sample_start_year, out_sample_start_month, 1)
out_sample_end <- lubridate::make_date(out_sample_end_year, out_sample_endt_month, 1)

idx_begin <- match(out_sample_begin, dates)
idx_end <- match(out_sample_end, dates)

EN34_out <- EN34[idx_begin:idx_end]
dates_out <- dates[idx_begin:idx_end]

x_out <- ts(EN34_out, start = c(out_sample_start_year, out_sample_start_month), frequency = 12)

# Combined in-sample + out-of-sample period
EN34_in_out <- c(EN34_in, EN34_out)
dates_in_out <- c(dates_in, dates_out)

# -----------------------------------------------------------------------------
# Time series plot (in-sample vs out-of-sample): Figure 1, top panel
# -----------------------------------------------------------------------------
pdf(here::here("../output/figures", "Figure1_enso_time_series_periodogram.pdf"), width = 10, height = 7)

showtext::showtext_begin()

par(mfrow = c(2, 1),
    family = "lora",
    mar = c(4, 4, 3, 1))

# Plotting time series
plot(dates_in_out, EN34_in_out,
     type = "l",
     lwd = 2,
     main = "Niño3.4 index",
     xlab = "",
     ylab = "Index",
     xaxt = "n")


# Custom x-axis ticks
ticks <- seq(as.Date("1985-01-01"),
             as.Date("2025-01-01"),
             by = "5 years")

axis(1, at = ticks, format(ticks, "%Y"))

# Reference line
ref <- out_sample_begin

ymax <- max(EN34_in_out)
ypos <- ymax * 0.99

text(ref, ypos, "Subsample 1", pos = 2, xpd = TRUE)
text(ref, ypos, "Subsample 2", pos = 4, xpd = TRUE)

abline(v = ref, lty = 2, col = "gray80")


# -----------------------------------------------------------------------------
# Periodogram (in-sample vs out-of-sample): Figure 1, bottom panel
# -----------------------------------------------------------------------------
# Compute spectrum
sp_in  <- spectrum(x_in,  spans = c(3, 3), plot = FALSE, log = "no", demean = TRUE, detrend = FALSE)
sp_out <- spectrum(x_out, spans = c(3, 3), plot = FALSE, log = "no", demean = TRUE, detrend = FALSE)

# Plot in-sample spectrum
plot(sp_in$freq, sp_in$spec,
     type = "l", lwd = 2, col = "blue",
     xlim = c(0, 2.3),
     ylim = range(c(sp_in$spec, sp_in$spec + 0.1)),
     xlab = "Cycles/year",
     ylab = "Spectral density",
     main = "Smoothed periodogram")

# Add out-of-sample spectrum
lines(sp_out$freq, sp_out$spec, lwd = 2, col = "red")

# Reference lines
refs <- c(12/6, 12/12, 12/18, 12/48)
labs <- c("6-month cycle", "12-month cycle", "18-month cycle", "48-month cycle")

ymax <- max(c(sp_in$spec, sp_in$spec))
ypos <- ymax * c(0.15, 0.6, 0.32, 0.99)

abline(v = refs, lty = 2, col = "gray80")

text(refs, ypos, labels = labs,
     pos = 3,
     xpd = TRUE)

# Legend
legend("topright",
       legend = c("Subsample 1", "Subsample 2"),
       col = c("blue", "red"),
       lwd = 2,
       bty = "n")

dev.off()

# -----------------------------------------------------------------------------
# HEGY test: Table A.1
# -----------------------------------------------------------------------------
# Test with seasonal dummies
hegy_res <- uroot::hegy.test(
  x_in,
  deterministic = c(1, 0, 1),  # constant, no trend, seasonal dummies
  lag.method = "AIC",
  maxlag = 12,
  pvalue = "RS"
)

print(hegy_res)

# Test without seasonal dummies
hegy_res <- uroot::hegy.test(
  x_in,
  deterministic = c(1, 0, 0),  # constant, no trend, no seasonal dummies
  lag.method = "AIC",
  maxlag = 12,
  pvalue = "RS"
)

print(hegy_res)

# -----------------------------------------------------------------------------
# Canova-Hansen test: Table A.2
# -----------------------------------------------------------------------------
# Annual cycle: pi/6
# Semiannual cycle: 2pi/6

# Test with trigonometrics (i.e. test I)
ch_res <- uroot::ch.test(
  x_in,
  type = "trigonometric",
  lag1 = TRUE,
  NW.order = 12, 
  pvalue = 'RS'
)

print(ch_res)

# Test with dummies (i.e. test II)
ch_res <- uroot::ch.test(
  x_in,
  type = "dummy",
  lag1 = TRUE,
  NW.order = 12, 
  pvalue = 'RS'
)

print(ch_res)

