library(EBImage)
library(png)
library(viridisLite)
library(grDevices)

setwd("./data")

overlay_mask <- function(img, mask, color = "#2F6EAD", alpha = 0.6) {
  
  # Clamp/normalize mask to [0,1]
  m <- pmin(pmax(mask, 0), 1)
  
  # Target overlay color as RGB [0,1]
  rgb_col <- grDevices::col2rgb(color) / 255
  r_t <- rgb_col[1]; g_t <- rgb_col[2]; b_t <- rgb_col[3]
  
  # Base grayscale replicated to RGB
  H <- dim(img)[1]; W <- dim(img)[2]
  R <- img[,,1]; G <- img[,,2]; B <- img[,,3]
  
  # Per-pixel effective alpha scales with mask value (so soft masks work naturally)
  a <- alpha * m
  
  # Alpha blend: out = (1 - a)*base + a*target_color
  R <- (1 - a) * R + a * r_t[1]
  G <- (1 - a) * G + a * g_t[1]
  B <- (1 - a) * B + a * b_t[1]

  
  # Stack to [H, W, 3], clamp to [0,1]
  out <- array(0, dim = c(H, W, 3))
  out[, 1:W, 1] <- pmin(pmax(R, 0), 1)
  out[, 1:W, 2] <- pmin(pmax(G, 0), 1)
  out[, 1:W, 3] <- pmin(pmax(B, 0), 1)
  out
}


to_rgb_image <- function(x01, palette = viridis(256)) {
  x01 <- pmin(pmax(x01, 0), 1)
  ramp <- colorRamp(palette)                  # function mapping [0,1] -> RGB(0..255)
  cols <- ramp(as.vector(x01))                # N x 3
  arr <- array(cols / 255, dim = c(nrow(x01), ncol(x01), 3))
  Image(arr, colormode = "Color")
}


names0 <- "ISIC_0000174"
path1 <- paste("test_uq", names0, sep = "/")

file_names <- list.files(path1)
file_names <- file_names[grep("c1.png", file_names)]

prob1 <- readPNG( paste(path1, file_names[1], sep = "/") )
H <- round(dim(prob1)[1])
W <- round(dim(prob1)[2])
prob1 <- pmin(pmax(prob1, 0), 1)
all_prob <- array(0, c(H, W, length(file_names)))
all_prob[,,1] <- prob1
for (i in 2:length(file_names)){
  prob_i <- readPNG( paste(path1, file_names[i], sep = "/") )
  prob_i <- pmin(pmax(prob_i, 0), 1)
  all_prob[,,i] <- prob_i
}

# (1) Average probability map across MC passes
avg_prob <- apply(all_prob, c(1, 2), mean) 

# (2) Predictive entropy H[E[p]] in **bits**
eps <- 1e-8
log2_safe <- function(x) log(x + eps, base = 2)
pred_entropy_bits <- -(avg_prob * log2_safe(avg_prob) +
                         (1 - avg_prob) * log2_safe(1 - avg_prob))   # H x W


# ---- Visualize (color) ----
avg_prob_rgb <- to_rgb_image(t(avg_prob), viridis(256))
predH_rgb    <- to_rgb_image(t(pred_entropy_bits),  inferno(256))
display(avg_prob_rgb, method = "raster")
display(predH_rgb,    method = "raster")


# Save to PNG (grayscale; values in [0,1])
writeImage(avg_prob_rgb,  paste0(path1, "/avg_probability_map.png"))
writeImage(predH_rgb, paste0(path1, "/predictive_entropy.png"))


img <- readJPEG( paste0(paste("imgs_test", names0, sep = "/"), ".jpg") )
mask_true <- readPNG( paste0(paste("masks_test", names0, sep = "/"), "_Segmentation.png") )
mask_pred <- readPNG( paste0(paste("masks_test_pred", names0, sep = "/"), "_OUT.png") )
img_mask_true <- overlay_mask(img, mask_true)
img_mask_pred <- overlay_mask(img, mask_pred, color = "#0F752E")
mask_disagree <-as.matrix(ifelse(mask_true != mask_pred, 1, 0))

writePNG(img, paste0(path1, "/img.png"))
writePNG(img_mask_true, paste0(path1, "/mask_true.png"))
writePNG(img_mask_pred, paste0(path1, "/mask_pred.png"))
writePNG(mask_disagree, paste0(path1, "/mask_disagree.png"))

