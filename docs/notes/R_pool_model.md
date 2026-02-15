# =========================================================
# 0. Load data from games.parquet
# =========================================================
library(arrow)
library(data.table)
suppressPackageStartupMessages(library(dplyr))

# read from your new file
x <- read_parquet("C:/GitHub/MTGA_DraftHelper/data/processed/games.parquet")
x <- as.data.table(x)

# basic checks (optional)
# str(x)
# head(x)

# ---------------------------------------------------------
# extract game-level columns used for base_p and calibration
# ---------------------------------------------------------
x2 <- x[, .(
  draft_id,
  won,
  user_n_games_bucket,
  user_game_win_rate_bucket,
  match_number
)]

# ---------------------------------------------------------
# build deck matrix X_deck (one row per draft, mean deck_*
# over games for that draft)
# ---------------------------------------------------------
library(data.table)

setDT(x)

## 1) identify matching deck_/sideboard_ columns
deck_cols      <- grep("^deck_",      names(x), value = TRUE)
sideboard_cols <- grep("^sideboard_", names(x), value = TRUE)

# assume 17Lands naming is consistent: deck_FOO vs sideboard_FOO
# align sideboard in the same order as deck_*
sideboard_from_deck <- sub("^deck_", "sideboard_", deck_cols)
if (!all(sideboard_from_deck %in% sideboard_cols)) {
  stop("Some deck_ columns have no matching sideboard_ column.")
}
sideboard_cols <- sideboard_from_deck  # now aligned with deck_cols

## 2) build pool_* = deck_* + sideboard_* row-wise
pool_cols <- sub("^deck_", "pool_", deck_cols)

for (i in seq_along(deck_cols)) {
  d_col  <- deck_cols[i]
  s_col  <- sideboard_cols[i]
  p_col  <- pool_cols[i]
  x[, (p_col) := as.numeric(get(d_col)) + as.numeric(get(s_col))]
}

## 3) aggregate per draft_id to get pool composition
# pool_* should be constant across games for a draft; max is robust
x3 <- x[, c("draft_id", pool_cols), with = FALSE]

deck_by_draft <- x3[
  ,
  c(
    list(n_games = .N),
    lapply(.SD, function(z) max(as.numeric(z), na.rm = TRUE))
    # use mean(...) instead of max(...) if you prefer
  ),
  by = .(draft_id),
  .SDcols = pool_cols
]
# Remove basic lands from the pool table
basic_lands <- c(
  "pool_island",
  "pool_swamp",
  "pool_forest",
  "pool_mountain",
  "pool_plains"
)

deck_by_draft <- deck_by_draft[, setdiff(names(deck_by_draft), basic_lands), with = FALSE]

## 4) feature matrix: pools, not decks
# Feature columns = pool_* columns except basic lands
pool_cols_final <- setdiff(pool_cols, basic_lands)

X_pool <- as.matrix(deck_by_draft[, ..pool_cols_final])
rownames(X_pool) <- deck_by_draft$draft_id


# =========================================================
# 1. CONFIG
# =========================================================
W <- 7L; L <- 3L
K_FOLDS <- 5
alpha <- 0.5; beta <- 0.5  # Jeffreys prior

# Optional: hand-tuned shrink strengths by games-played bucket (helps bias)
K_MAP <- c(`1`=15, `5`=30, `10`=55, `50`=75, `100`=95, `500`=115, `1000`=125)


# =========================================================
# 2. HELPERS
# =========================================================
clip01 <- function(p, eps=1e-9) pmin(pmax(p, eps), 1 - eps)
invlog <- function(z) 1/(1+exp(-z))

coerce_won01 <- function(x) {
  if (is.logical(x)) as.integer(x)
  else if (is.numeric(x)) as.integer(x != 0)
  else if (is.character(x)) as.integer(tolower(x) %in% c("true","t","1","yes"))
  else stop("'won' must be logical/numeric/character")
}

derive_k_by_gp <- function(gp_vec, k_hi = 80, k_lo = 15) {
  lev <- sort(unique(as.character(gp_vec)))
  lev_num <- suppressWarnings(as.numeric(lev))
  ord <- if (all(!is.na(lev_num))) order(lev_num) else order(lev)
  lev <- lev[ord]
  t <- seq(0,1,length.out=length(lev))
  setNames((1-t)*k_hi + t*k_lo, lev)
}

# posterior mean per draft respecting stop rule (for QA only;
# NOT used to fit calibration to MLE)
posterior_mean_by_draft <- function(df, W=7L, L=3L, alpha=.5, beta=.5) {
  df$won01 <- coerce_won01(df$won)
  drafts <- by(df[, c("draft_id","match_number","won01")],
               df$draft_id,
               function(dd) {
                 dd <- dd[order(dd$match_number, na.last=TRUE), ]
                 cw <- cumsum(dd$won01)
                 cl <- cumsum(1L - dd$won01)
                 idx <- which(cw >= W | cl >= L)[1]
                 if (!is.na(idx)) {
                   w_stop <- cw[idx]; l_stop <- cl[idx]
                   if (l_stop >= L) {
                     a <- alpha + w_stop
                     b <- beta + L
                   } else {
                     a <- alpha + W
                     b <- beta + l_stop
                   }
                 } else {
                   w_stop <- sum(dd$won01)
                   l_stop <- sum(1L - dd$won01)
                   a <- alpha + w_stop
                   b <- beta + l_stop
                 }
                 data.frame(
                   draft_id    = dd$draft_id[1],
                   w_stop      = w_stop,
                   l_stop      = l_stop,
                   p_post_draft = clip01(a/(a+b))
                 )
               })
  do.call(rbind, drafts)
}


# =========================================================
# 3. OOF base_p (two buckets, hierarchical shrink)
# =========================================================
oof_base_p_two_buckets_hier <- function(df,
                                        draft_col = "draft_id", won_col = "won",
                                        wr_col = "user_game_win_rate_bucket",
                                        gp_col = "user_n_games_bucket",
                                        K = 5,
                                        alpha = .5, beta = .5,
                                        k_by_gp = NULL, default_k = 40,
                                        seed = 1) {
  set.seed(seed)
  draft <- as.character(df[[draft_col]])
  won01 <- coerce_won01(df[[won_col]])
  wr    <- as.character(df[[wr_col]])
  gp    <- as.character(df[[gp_col]])
  key   <- paste(wr, gp, sep="||")
  
  if (is.null(k_by_gp)) k_by_gp <- derive_k_by_gp(gp)
  get_k <- function(g) {
    out <- k_by_gp[as.character(g)]
    ifelse(is.na(out), default_k, as.numeric(out))
  }
  
  drafts  <- unique(draft)
  fold_id <- setNames(sample(rep(1:K, length.out=length(drafts))), drafts)
  fold    <- fold_id[draft]
  
  base_hat <- numeric(nrow(df))
  
  for (k in 1:K) {
    tr <- fold != k
    va <- fold == k
    
    # WR-only marginal
    W_wr <- rowsum(won01[tr], group = wr[tr], reorder = FALSE)
    N_wr <- rowsum(rep.int(1L, sum(tr)), group = wr[tr], reorder = FALSE)
    base_wr <- as.numeric(W_wr + alpha) / as.numeric(N_wr + alpha + beta)
    names(base_wr) <- rownames(W_wr)
    
    # joint (wr,gp)
    W_joint <- rowsum(won01[tr], group = key[tr], reorder = FALSE)
    N_joint <- rowsum(rep.int(1L, sum(tr)), group = key[tr], reorder = FALSE)
    
    joint_keys <- rownames(W_joint)
    wr_of_key  <- sub("\\|\\|.*$", "", joint_keys)
    gp_of_key  <- sub("^.*\\|\\|",  "", joint_keys)
    m0_wr      <- base_wr[wr_of_key]
    k_gp       <- sapply(gp_of_key, get_k)
    
    base_joint <- (as.numeric(W_joint) + k_gp*m0_wr + alpha) /
      (as.numeric(N_joint) + k_gp + alpha + beta)
    names(base_joint) <- joint_keys
    
    base_global <- (sum(W_joint) + alpha) / (sum(N_joint) + alpha + beta)
    
    k_va <- key[va]
    wr_va <- wr[va]
    bh   <- base_joint[k_va]
    miss <- is.na(bh)
    if (any(miss)) {
      bh[miss] <- base_wr[wr_va[miss]]
      miss <- is.na(bh)
      if (any(miss)) bh[miss] <- base_global
    }
    base_hat[va] <- bh
  }
  
  clip01(base_hat)
}


# =========================================================
# 4. grouped logistic calibration vs MLE
# =========================================================
# z = logit(base_p) + θ0 + θ1*logit(base_p) + γ_gp + φ_gp*logit(base_p)
fit_basep_logit_cal_grouped <- function(d_draft) {
  s   <- qlogis(clip01(d_draft$base_p))
  gp  <- factor(d_draft$gp_bucket)
  Xgp <- model.matrix(~ gp - 1)   # one-hot; first level baseline
  G   <- ncol(Xgp)
  A <- d_draft$A
  B <- d_draft$B
  
  nll <- function(p) {
    p <- clip01(p)
    -mean(A*log(p) + B*log(1-p))
  }
  
  n_params <- 2 + (G-1) + (G-1)
  par0 <- rep(0, n_params)
  
  obj <- function(par) {
    theta0 <- par[1]
    theta1 <- par[2]
    gamma  <- c(0, par[3:(2+G-1)])
    phi    <- c(0, par[(2+G):(2+2*G-2)])
    z <- s + theta0 + theta1*s +
      as.numeric(Xgp %*% gamma) +
      as.numeric((Xgp %*% phi) * s)
    nll(invlog(z))
  }
  
  fit <- optim(par0, obj, method = "BFGS")
  list(
    theta0 = fit$par[1],
    theta1 = fit$par[2],
    gamma  = setNames(c(0, fit$par[3:(2+G-1)]), colnames(Xgp)),
    phi    = setNames(c(0, fit$par[(2+G):(2+2*G-2)]), colnames(Xgp)),
    gp_levels = levels(gp)
  )
}

apply_basep_logit_cal_grouped <- function(base_p, gp_bucket, cal) {
  s  <- qlogis(clip01(base_p))
  gp <- factor(gp_bucket, levels = cal$gp_levels)
  X  <- model.matrix(~ gp - 1)
  gamma <- cal$gamma[colnames(X)]; gamma[is.na(gamma)] <- 0
  phi   <- cal$phi  [colnames(X)]; phi  [is.na(phi)]   <- 0
  z <- s + cal$theta0 + cal$theta1*s +
    as.numeric(X %*% gamma) +
    as.numeric((X %*% phi) * s)
  clip01(invlog(z))
}


# =========================================================
# 5. diagnostics (slope/intercept, ECE, bucket residuals)
# =========================================================
basep_report <- function(x2, basep_col, nbins = 12) {
  stopifnot(all(c("draft_id","w_stop","l_stop","user_n_games_bucket") %in% names(x2)))
  # 1) one row per draft
  d <- x2 %>%
    group_by(draft_id) %>%
    summarise(
      base_p   = mean(.data[[basep_col]], na.rm = TRUE),
      w_stop   = first(w_stop),
      l_stop   = first(l_stop),
      gp_bucket= first(user_n_games_bucket),
      .groups  = "drop"
    ) %>%
    mutate(
      A = ifelse(w_stop >= W, W, w_stop),
      B = ifelse(l_stop >= L, L, l_stop),
      p_mle = A/(A+B),
      w = A+B
    )
  
  # 2) weighted calibration line: p_mle ~ base_p
  reg <- summary(lm(p_mle ~ base_p, data = d, weights = w))$coef
  
  # 3) reliability bins (ECE) vs MLE
  br <- unique(quantile(d$base_p, probs = seq(0,1,length.out = nbins+1), na.rm = TRUE))
  if (length(br) < 3) br <- pretty(range(d$base_p, na.rm = TRUE), n = nbins)
  d$bin <- cut(d$base_p, breaks = br, include.lowest = TRUE, ordered_result = TRUE)
  
  p_bins <- d %>%
    group_by(bin) %>%
    summarise(
      p_mean = mean(base_p),
      A = sum(A),
      B = sum(B),
      .groups="drop"
    ) %>%
    mutate(
      p_mle = A/(A+B),
      w = A+B
    )
  ECE <- with(p_bins, weighted.mean(abs(p_mean - p_mle), w))
  
  # 4) per-games-played-bucket residuals
  by_gp <- d %>%
    group_by(gp_bucket) %>%
    summarise(
      n = n(),
      mean_delta = mean(p_mle - base_p),
      se_delta   = sd(p_mle - base_p)/sqrt(n),
      .groups="drop"
    ) %>%
    arrange(gp_bucket)
  
  list(regression = reg, ECE = ECE, by_gp = by_gp, p_bins = p_bins, drafts = d)
}


# =========================================================
# 6. PIPELINE on games.parquet
# =========================================================

# 0) stop stats & posterior mean per draft (QA only)
post_tbl <- posterior_mean_by_draft(x2, W=W, L=L, alpha=alpha, beta=beta)
x2 <- merge(x2, post_tbl, by="draft_id", all.x=TRUE, sort=FALSE)

# 1) OOF base_p (two buckets with hierarchical shrink)
x2$base_p <- oof_base_p_two_buckets_hier(
  x2,
  draft_col="draft_id", won_col="won",
  wr_col="user_game_win_rate_bucket",
  gp_col="user_n_games_bucket",
  K=K_FOLDS, alpha=alpha, beta=beta,
  k_by_gp = K_MAP, default_k = 40
)

# 2) draft-level table for calibration to MLE
d_draft <- within(aggregate(base_p ~ draft_id, x2, mean), {
  w_stop <- aggregate(w_stop ~ draft_id, x2, `[`, 1)$w_stop
  l_stop <- aggregate(l_stop ~ draft_id, x2, `[`, 1)$l_stop
  gp_bucket <- aggregate(user_n_games_bucket ~ draft_id, x2, `[`, 1)$user_n_games_bucket
  A <- ifelse(w_stop >= W, W, w_stop)
  B <- ifelse(l_stop >= L, L, l_stop)
  p_mle <- A/(A+B)
})
names(d_draft)[2] <- "base_p"
d_draft <- d_draft[, c("draft_id","base_p","w_stop","l_stop","gp_bucket")]
d_draft$A <- ifelse(d_draft$w_stop >= W, W, d_draft$w_stop)
d_draft$B <- ifelse(d_draft$l_stop >= L, L, d_draft$l_stop)

# 3) fit grouped logistic calibration (vs MLE)
cal <- fit_basep_logit_cal_grouped(d_draft)

# 4) apply calibration row-wise -> base_p_cal
x2$base_p_cal <- apply_basep_logit_cal_grouped(
  base_p    = x2$base_p,
  gp_bucket = x2$user_n_games_bucket,
  cal       = cal
)

# 5) QA: expect intercept ~0, slope ~1; low ECE; near-zero bucket residuals
rep_raw <- basep_report(x2, basep_col = "base_p")
rep_raw$regression
rep_raw$ECE
rep_raw$by_gp

rep_cal <- basep_report(x2, basep_col = "base_p_cal")
rep_cal$regression
rep_cal$ECE
rep_cal$by_gp

# 6) save artifacts for reuse
saveRDS(x2,     "FIN_gamedata_x2_with_basep.rds")
saveRDS(X_deck, "FIN_pool_X_deck.rds")
saveRDS(
  list(
    K_MAP  = K_MAP,
    cal    = cal,
    W      = W,
    L      = L,
    alpha  = alpha,
    beta   = beta
  ),
  "basep_artifacts.rds"
)

## =========================================================
## POOL-BASED XGBOOST "POOL EFFECT" MODEL + DIAGNOSTICS
## =========================================================

library(data.table)
library(xgboost)
library(ggplot2)

setDT(x2)

clip01 <- function(p, eps=1e-9) pmin(pmax(p, eps), 1 - eps)
invlog  <- function(z) 1/(1+exp(-z))
run_nll <- function(p, A, B) {
  p <- clip01(p)
  -mean(A*log(p) + B*log(1-p))
}

## --- posterior mean per draft under 7W/3L (Jeffreys) ---
## (use game-level won + match_number; if you already defined an event-level
##  version earlier, skip this and use that instead)
posterior_mean_by_draft <- function(df, W = 7L, L = 3L, a = .5, b = .5) {
  df[, won01 := as.integer(as.logical(won))]
  df[order(draft_id, match_number),
     {
       cw <- cumsum(won01); cl <- cumsum(1L - won01)
       idx <- which(cw >= W | cl >= L)[1]
       if (length(idx) && !is.na(idx)) {
         w <- cw[idx]; l <- cl[idx]
         A <- if (w >= W) a + W else a + w
         B <- if (l >= L) b + L else b + l
       } else {
         w <- sum(won01); l <- sum(1L - won01)
         A <- a + w; B <- b + l
       }
       .(w_stop = w, l_stop = l, p_post_draft = A/(A + B))
     },
     by = draft_id]
}

## --- reliability / ECE vs bin MLE ---
reliability_ece <- function(p, A, B, nbins = 12, title = "Reliability") {
  p <- as.numeric(p); A <- as.numeric(A); B <- as.numeric(B)
  br <- unique(quantile(p, probs = seq(0, 1, length.out = nbins + 1)))
  if (length(br) < 3) br <- pretty(range(p), n = nbins)
  bin <- cut(p, breaks = br, include.lowest = TRUE)
  tab <- data.table(bin, p, A, B)[,
                                  .(p_mean = mean(p), A = sum(A), B = sum(B)),
                                  by = bin][
                                    , `:=`(p_mle = A/(A + B), w = A + B)]
  
  ece <- tab[, weighted.mean(abs(p_mean - p_mle), w)]
  plt <- ggplot(tab, aes(p_mean, p_mle, size = w)) +
    geom_point(alpha = .9) +
    geom_abline(slope = 1, intercept = 0, linetype = 2) +
    labs(title = title, x = "mean predicted p", y = "bin MLE p") +
    theme_minimal() +
    theme(plot.background  = element_rect(fill = "white", colour = NA),
          panel.background = element_rect(fill = "white", colour = NA))
  list(ece = ece, plot = plt, table = tab)
}

## --- uplift within base_p deciles (ranking sanity) ---
uplift_by_base <- function(base_p, s_hat, A, B, nbins = 10L) {
  dt <- data.table(base_p, s_hat, A, B)
  dt[, bin := cut(base_p,
                  quantile(base_p, probs = seq(0, 1, length.out = nbins + 1)),
                  include.lowest = TRUE)]
  out <- dt[, {
    hi <- s_hat >= quantile(s_hat, .8)
    lo <- s_hat <= quantile(s_hat, .2)
    A_hi <- sum(A[hi]); B_hi <- sum(B[hi])
    A_lo <- sum(A[lo]); B_lo <- sum(B[lo])
    list(
      uplift = (A_hi/(A_hi + B_hi)) - (A_lo/(A_lo + B_lo)),
      w      = (A_hi + B_hi) + (A_lo + B_lo)
    )
  }, by = bin]
  out[, weighted.mean(uplift, w)]
}

## ================= build draft-level frame & merge POOL features =================

# 1) posterior means + stop stats
# choose baseline column
# 1. posterior means (for stop rule)
post_tbl <- posterior_mean_by_draft(x2)

# 2. compute OOF hierarchical base WR
x2$base_p <- oof_base_p_two_buckets_hier(
  x2,
  draft_col="draft_id", won_col="won",
  wr_col="user_game_win_rate_bucket",
  gp_col="user_n_games_bucket",
  K = 5,
  alpha = 0.5, beta = 0.5,
  k_by_gp = K_MAP,    # or derive_k_by_gp()
  default_k = 40
)

# 3. build draft-level summary for logistic calibration
d_draft <- within(aggregate(base_p ~ draft_id, x2, mean), {
  # add stop-rule data
  w_stop <- aggregate(w_stop ~ draft_id, post_tbl, `[`, 1)$w_stop
  l_stop <- aggregate(l_stop ~ draft_id, post_tbl, `[`, 1)$l_stop
  gp_bucket <- aggregate(user_n_games_bucket ~ draft_id, x2, `[`, 1)$user_n_games_bucket
  
  A <- pmin(w_stop, 7L)
  B <- pmin(l_stop, 3L)
  p_mle <- A / (A + B)
})

d_draft$A <- pmin(d_draft$w_stop, 7L)
d_draft$B <- pmin(d_draft$l_stop, 3L)

# 4. fit grouped logistic calibration
cal <- fit_basep_logit_cal_grouped(d_draft)

# 5. apply calibration
x2$base_p_grpcal <- apply_basep_logit_cal_grouped(
  base_p    = x2$base_p,
  gp_bucket = x2$user_n_games_bucket,
  cal       = cal
)

# 2) baseline per draft: mean calibrated base_p + games-played bucket
# choose baseline column
base_col <- if ("base_p_grpcal" %in% names(x2)) {
  "base_p_grpcal"
} else if ("base_p" %in% names(x2)) {
  "base_p"
} else {
  stop("Neither base_p_grpcal nor base_p found in x2.")
}

post_tbl <- posterior_mean_by_draft(x2)

draft_base <- x2[, .(
  base_p   = mean(get(base_col)),
  gp_bucket = first(user_n_games_bucket)
), by = draft_id][post_tbl, on = "draft_id"]



# 3) stop-rule weights & MLE (for eval)
draft_base[, `:=`(
  A     = pmin(w_stop, 7L),
  B     = pmin(l_stop, 3L),
  p_mle = pmin(w_stop, 7) / (pmin(w_stop, 7) + pmin(l_stop, 3))
)]

# 4) pool features (rows = draft_id; columns = pool_* counts)
X_df <- as.data.table(X_pool)
X_df[, draft_id := rownames(X_pool)]
setcolorder(X_df, c("draft_id", setdiff(names(X_df), "draft_id")))

# 5) inner join: only drafts present in both x2 and pool matrix
D <- merge(draft_base, X_df, by = "draft_id", all = FALSE)

# 6) feature matrix and target
pool_feature_cols <- setdiff(
  names(D),
  c("draft_id", "base_p", "gp_bucket", "w_stop", "l_stop",
    "p_post_draft", "A", "B", "p_mle")
)

X <- as.matrix(cbind(base_p = D$base_p, D[, ..pool_feature_cols]))
y <- D$p_post_draft - D$base_p     # surrogate Δp (pool effect)
w <- D$A + D$B                     # effective weight
ids <- D$draft_id
base_p_vec <- D$base_p

## ================= train/valid/test split (stratified by base_p) =================

set.seed(123)
strata   <- cut(base_p_vec,
                quantile(base_p_vec, probs = seq(0, 1, 0.1)),
                include.lowest = TRUE)
split_idx <- split(seq_along(ids), strata)
pick <- function(v, frac) sample(v, ceiling(length(v) * frac))

val_idx  <- unlist(lapply(split_idx, pick, frac = 0.15))
rest     <- setdiff(seq_along(ids), val_idx)
test_idx <- unlist(lapply(split(rest, strata[rest]), pick, frac = 0.15))
train_idx <- setdiff(rest, test_idx)

## ================= XGBoost on surrogate Δp =================

dtrain <- xgb.DMatrix(X[train_idx, ], label = y[train_idx], weight = w[train_idx])
dvalid <- xgb.DMatrix(X[val_idx,   ], label = y[val_idx],   weight = w[val_idx])

params <- list(
  objective        = "reg:squarederror",
  eta              = 0.05,
  max_depth        = 6,
  min_child_weight = 2,
  subsample        = 0.9,
  colsample_bytree = 1.0,
  lambda           = 0.5,
  tree_method      = "hist",
  eval_metric      = "rmse"
)

fit_xgb <- xgb.train(
  params = params,
  data   = dtrain,
  nrounds = 5000,
  watchlist = list(train = dtrain, valid = dvalid),
  early_stopping_rounds = 150,
  verbose = 1
)

# surrogate predictions ŝ
s_val  <- as.numeric(predict(fit_xgb, X[val_idx, ]))
s_test <- as.numeric(predict(fit_xgb, X[test_idx, ]))

## ================= logistic calibration on VALID (2 params) =================

nll_theta <- function(th, base_p, s_hat, A, B) {
  z <- qlogis(clip01(base_p)) + th[1] + th[2] * s_hat
  p <- clip01(invlog(z))
  run_nll(p, A, B)
}

th <- optim(
  c(0, 1), nll_theta,
  base_p = base_p_vec[val_idx],
  s_hat  = s_val,
  A      = D$A[val_idx],
  B      = D$B[val_idx],
  method = "BFGS"
)$par

theta0 <- th[1]; theta1 <- th[2]

# calibrated probabilities
p_val_model  <- clip01(invlog(qlogis(clip01(base_p_vec[val_idx])) + theta0 + theta1 * s_val))
p_test_model <- clip01(invlog(qlogis(clip01(base_p_vec[test_idx])) + theta0 + theta1 * s_test))
p_val_base   <- clip01(base_p_vec[val_idx])
p_test_base  <- clip01(base_p_vec[test_idx])

## ================= evaluation: NLL, reliability, uplift =================

# NLL
nll_val_model <- run_nll(p_val_model,  D$A[val_idx],  D$B[val_idx])
nll_val_base  <- run_nll(p_val_base,   D$A[val_idx],  D$B[val_idx])
nll_tst_model <- run_nll(p_test_model, D$A[test_idx], D$B[test_idx])
nll_tst_base  <- run_nll(p_test_base,  D$A[test_idx], D$B[test_idx])

cat("NLL VALID (model/base):", nll_val_model, nll_val_base, "\n")
cat("NLL TEST  (model/base):", nll_tst_model, nll_tst_base, "\n")
c(rel_gain_val = 1 - nll_val_model/nll_val_base,
  rel_gain_tst = 1 - nll_tst_model/nll_tst_base)

# Reliability (ECE)
rel_val <- reliability_ece(p_val_model,  D$A[val_idx],  D$B[val_idx],
                           nbins = 12, title = "Reliability (VALID)")
rel_tst <- reliability_ece(p_test_model, D$A[test_idx], D$B[test_idx],
                           nbins = 12, title = "Reliability (TEST)")
rel_val$ece; rel_tst$ece
# print(rel_val$plot); print(rel_tst$plot)  # if you want to see the diagrams

# Uplift (ranking within base_p deciles)
upl_val <- uplift_by_base(base_p_vec[val_idx], s_val,  D$A[val_idx],  D$B[val_idx])
upl_tst <- uplift_by_base(base_p_vec[test_idx], s_test, D$A[test_idx], D$B[test_idx])
c(uplift_valid = upl_val, uplift_test = upl_tst)

## ================= deck-effect calibration plots (Δp vs observed) =================

deck_effect_calibration <- function(base_p, s_hat, A, B, p_mle, p_post = NULL,
                                    theta0, theta1, nbins = 12, label = "SPLIT") {
  stopifnot(length(base_p) == length(s_hat),
            length(A) == length(B),
            length(A) == length(p_mle))
  
  p_hat  <- clip01(invlog(qlogis(clip01(base_p)) + theta0 + theta1 * s_hat))
  de_hat <- p_hat - base_p
  
  de_obs_mle <- p_mle - base_p
  if (!is.null(p_post)) de_obs_post <- p_post - base_p
  
  w <- A + B
  reg_mle <- summary(lm(de_obs_mle ~ de_hat, weights = w))$coef
  
  br <- unique(quantile(de_hat, probs = seq(0, 1, length.out = nbins + 1), na.rm = TRUE))
  if (length(br) < 3) br <- pretty(range(de_hat, na.rm = TRUE), n = nbins)
  bin <- cut(de_hat, breaks = br, include.lowest = TRUE, ordered_result = TRUE)
  dt  <- data.table(bin, de_hat, de_obs_mle, w)
  tab <- dt[, .(x_pred = weighted.mean(de_hat, w),
                y_obs  = weighted.mean(de_obs_mle, w),
                w = sum(w)), by = bin]
  
  ece_delta <- tab[, weighted.mean(abs(x_pred - y_obs), w)]
  
  p <- ggplot(tab, aes(x_pred, y_obs, size = w)) +
    geom_point(alpha = 0.95) +
    geom_abline(slope = 1, intercept = 0, linetype = 2) +
    labs(title = paste0("Pool-effect calibration (", label, ")"),
         x = "Predicted Δp (bin mean)", y = "Observed Δp (bin mean, MLE)") +
    theme_minimal() +
    theme(plot.background  = element_rect(fill = "white", colour = NA),
          panel.background = element_rect(fill = "white", colour = NA))
  
  out <- list(
    plot        = p,
    ece_delta   = ece_delta,
    regression_mle = reg_mle,
    table       = tab,
    de_hat      = de_hat,
    de_obs_mle  = de_obs_mle
  )
  
  if (!is.null(p_post)) {
    de_obs_post <- p_post - base_p
    reg_post <- summary(lm(de_obs_post ~ de_hat, weights = w))$coef
    dtp <- data.table(bin, de_hat, de_obs_post, w)
    tabp <- dtp[, .(x_pred = weighted.mean(de_hat, w),
                    y_obs  = weighted.mean(de_obs_post, w),
                    w = sum(w)), by = bin]
    ece_delta_post <- tabp[, weighted.mean(abs(x_pred - y_obs), w)]
    p2 <- ggplot(tabp, aes(x_pred, y_obs, size = w)) +
      geom_point(alpha = 0.95) +
      geom_abline(slope = 1, intercept = 0, linetype = 2) +
      labs(title = paste0("Pool-effect calibration vs Posterior (", label, ")"),
           x = "Predicted Δp (bin mean)", y = "Observed Δp (bin mean, posterior)") +
      theme_minimal() +
      theme(plot.background  = element_rect(fill = "white", colour = NA),
            panel.background = element_rect(fill = "white", colour = NA))
    out$plot_post       <- p2
    out$ece_delta_post  <- ece_delta_post
    out$regression_post <- reg_post
    out$table_post      <- tabp
  }
  out
}

# VALID and TEST diagnostics
val <- deck_effect_calibration(
  base_p = D$base_p[val_idx],
  s_hat  = s_val,
  A      = D$A[val_idx],
  B      = D$B[val_idx],
  p_mle  = D$p_mle[val_idx],
  p_post = D$p_post_draft[val_idx],
  theta0 = theta0,
  theta1 = theta1,
  nbins  = 12,
  label  = "VALID"
)
print(val$plot)

tst <- deck_effect_calibration(
  base_p = D$base_p[test_idx],
  s_hat  = s_test,
  A      = D$A[test_idx],
  B      = D$B[test_idx],
  p_mle  = D$p_mle[test_idx],
  p_post = D$p_post_draft[test_idx],
  theta0 = theta0,
  theta1 = theta1,
  nbins  = 100,
  label  = "TEST"
)
print(tst$plot)

## ================= final pooled model on ALL data =================

D_all <- as.data.table(D)  # already full

drop <- c("draft_id","gp_bucket","w_stop","l_stop",
          "p_post_draft","A","B","p_mle")
feature_cols_all <- setdiff(names(D_all), drop)

X_all <- as.matrix(D_all[, ..feature_cols_all])
y_all <- D_all$p_post_draft - D_all$base_p
w_all <- D_all$A + D_all$B

dm <- xgb.DMatrix(data = X_all, label = y_all, weight = w_all)

params_final <- list(
  objective        = "reg:squarederror",
  eval_metric      = "rmse",
  eta              = 0.08,
  max_depth        = 6,
  min_child_weight = 2,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  reg_alpha        = 0,
  reg_lambda       = 1
)

cv <- xgb.cv(
  params = params_final,
  data   = dm,
  nrounds = 4000,
  nfold   = 5,
  early_stopping_rounds = 200,
  verbose = 0
)
best_nrounds <- cv$best_iteration
if (is.null(best_nrounds) || best_nrounds <= 0) {
  best_nrounds <- which.min(cv$evaluation_log$test_rmse_mean)
}
cat(sprintf("CV picked %d rounds (test_rmse=%.4f)\n",
            best_nrounds,
            cv$evaluation_log$test_rmse_mean[best_nrounds]))

set.seed(123)
fit_final <- xgb.train(
  params  = params_final,
  data    = dm,
  nrounds = best_nrounds,
  verbose = 0
)

dir.create("models", showWarnings = FALSE)
xgb.save(fit_final, "models/pool_effect_xgb.json")

meta <- list(
  feature_names = colnames(X_all),
  pool_cols     = pool_feature_cols,
  params        = params_final,
  nrounds       = best_nrounds,
  nrows         = nrow(X_all),
  target        = "p_post_draft - base_p",
  weights       = "A + B",
  timestamp     = as.character(Sys.time())
)
saveRDS(meta, "models/pool_effect_xgb_meta.rds")

imp <- xgb.importance(model = fit_final)
data.table::fwrite(imp, "models/pool_effect_xgb_importance.csv")

# sanity: in-sample calibration stats
pred_all <- predict(fit_final, dm)
fit_cal  <- lm(y_all ~ pred_all, weights = w_all)
cat(sprintf("final(all): slope=%.3f, intercept=%.3f, R2=%.3f, RMSE=%.3f\n",
            coef(fit_cal)[2], coef(fit_cal)[1],
            summary(fit_cal)$r.squared,
            sqrt(weighted.mean((y_all - pred_all)^2, w_all))))

