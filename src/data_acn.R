# Téléchargement et cache des sessions ACN-Data (Caltech).
# Token : variable ACN_API_TOKEN (fichier .Renviron à la racine, ignoré par git).
# Usage : source("src/data_acn.R"); d <- load_acn()            # tous les sites
#         Rscript src/data_acn.R                                # pré-remplit le cache
library(httr)
library(jsonlite)

ACN_SITES <- c("caltech", "jpl", "office001")
ACN_BASE  <- "https://ev.caltech.edu/api/v1/"

.acn_date <- function(x) as.POSIXct(x, format = "%a, %d %b %Y %H:%M:%S", tz = "GMT")

# Dernière saisie utilisateur (userInputs est une liste d'éditions successives)
.last_input <- function(ui, field) {
  if (is.null(ui) || length(ui) == 0) return(NA)
  v <- ui[[length(ui)]][[field]]
  if (is.null(v)) NA else v
}

# Période par défaut : sept. 2018 (userInputs généralisés) à mars 2020 (avant le COVID).
# Le filtre "where" est indispensable : sans lui la pagination profonde prend ~40 s par page.
ACN_START <- "Sat, 01 Sep 2018 00:00:00 GMT"
ACN_END   <- "Tue, 31 Mar 2020 23:59:59 GMT"

fetch_acn <- function(site, start = ACN_START, end = ACN_END, token = Sys.getenv("ACN_API_TOKEN")) {
  if (!nzchar(token)) stop("ACN_API_TOKEN absent : ajoute-le dans .Renviron")
  where <- sprintf('connectionTime>="%s" and connectionTime<="%s"', start, end)
  pages <- list(); page <- 1
  repeat {
    r <- RETRY("GET", paste0(ACN_BASE, "sessions/", site), authenticate(token, ""),
               query = list(where = where, max_results = 100, page = page), times = 5, pause_base = 2)
    stop_for_status(r)
    p <- fromJSON(content(r, "text", encoding = "UTF-8"), simplifyVector = FALSE)
    pages[[page]] <- p$`_items`
    if (is.null(p$`_links`$`next`)) break
    page <- page + 1
  }
  items <- do.call(c, pages)
  num <- function(x) if (is.null(x)) NA_real_ else as.numeric(x)
  chr <- function(x) if (is.null(x)) NA_character_ else as.character(x)
  data.frame(
    site               = site,
    sessionID          = vapply(items, function(i) chr(i$sessionID), ""),
    stationID          = vapply(items, function(i) chr(i$stationID), ""),
    spaceID            = vapply(items, function(i) chr(i$spaceID), ""),
    userID             = vapply(items, function(i) chr(i$userID), ""),
    connectionTime     = .acn_date(vapply(items, function(i) chr(i$connectionTime), "")),
    disconnectTime     = .acn_date(vapply(items, function(i) chr(i$disconnectTime), "")),
    doneChargingTime   = .acn_date(vapply(items, function(i) chr(i$doneChargingTime), "")),
    kWhDelivered       = vapply(items, function(i) num(i$kWhDelivered), 0),
    timezone           = vapply(items, function(i) chr(i$timezone), ""),
    kWhRequested       = vapply(items, function(i) num(.last_input(i$userInputs, "kWhRequested")), 0),
    minutesAvailable   = vapply(items, function(i) num(.last_input(i$userInputs, "minutesAvailable")), 0),
    requestedDeparture = .acn_date(vapply(items, function(i) chr(.last_input(i$userInputs, "requestedDeparture")), "")),
    milesRequested     = vapply(items, function(i) num(.last_input(i$userInputs, "milesRequested")), 0),
    WhPerMile          = vapply(items, function(i) num(.last_input(i$userInputs, "WhPerMile")), 0),
    paymentRequired    = vapply(items, function(i) as.logical(chr(.last_input(i$userInputs, "paymentRequired"))), NA),
    stringsAsFactors = FALSE
  )
}

# Charge un ou plusieurs sites, avec cache data/acn_<site>.rds
load_acn <- function(sites = ACN_SITES, cache_dir = "data", refresh = FALSE) {
  dir.create(cache_dir, showWarnings = FALSE)
  do.call(rbind, lapply(sites, function(s) {
    f <- file.path(cache_dir, paste0("acn_", s, ".rds"))
    if (refresh || !file.exists(f)) { message("Téléchargement ", s, "..."); saveRDS(fetch_acn(s), f) }
    readRDS(f)
  }))
}

if (sys.nframe() == 0) {
  d <- load_acn()
  print(table(d$site)); str(d)
}
