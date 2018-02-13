# load zipcode data
data(zipcode)

# generate the rental price index from Zillow data
importZillowData <- function() {
  zillow_file <- file.path("data/Zip_MedianRentalPrice_AllHomes.csv")
  
  rentalIndex <- read_csv(file = zillow_file, 
                          col_types = cols(RegionName = col_character()))
  
  rentalIndex <- rentalIndex %>%
    rename(rentalindex = `2017-12`) %>%
    rename(zip = RegionName)
  
  rentalIndex <- rentalIndex %>%
    select(zip, rentalindex) %>%
    right_join(zipcode, by = c("zip" = "zip")) %>%
    drop_na()
  return(rentalIndex)
}

rentalIndex <- importZillowData()

# get the rental index for a given zip
getRentalIndex <- function(zipcodes) {
  val <- vector(length = length(zipcodes))
  for (i in 1:length(zipcodes)) {
    nearestZipWithData <- getNearestZip(charZip)
    x <- rentalIndex %>%
      filter(zip %in% nearestZipWithData) %>%
      select(rentalIndex) %>%
      as.numeric()
    val[i] <- x
  }
  return(val)
}

# get the coordinates of the zipcode
getCoordFromZip <- function(charZip) {
  coord <- zipcode %>%
    filter(zip %in% charZip) %>%
    select(longitude, latitude) %>%
    as.matrix()
  return(coord)
}

# get the state based on the zipcode
getStateFromZip <- function(charZip) {
  state <- zipcode %>%
    filter(zip %in% charZip) %>%
    select(state) %>%
    as.character()
  return(state)
}
 
# get the city based on the zipcode
getCityFromZip <- function(charZip) {
  city <- zipcode %>%
    filter(zip %in% charZip) %>%
    select(city) %>%
    as.character()
  return(city)
}

# get the zipcode closest to the query zipcode for which we have 
# data populated in the Rental Price Index
getNearestZip <- function(charZip) {
  zipWithData <- rentalIndex %>%
    select(zip)

  query_city <- getCityFromZip(charZip)

  query_space <- zipcode %>%
    filter(city %in% query_city & zip %in% zipWithData$zip)
  
  query_points <- query_space %>%
    select(longitude, latitude) %>%
    as.matrix()
  
  target_coord <- getCoordFromZip(charZip)
  m <- distm(target_coord, query_points)
  i <- which.min(m)
  return(query_space$zip[i])
}


