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
getRentalIndex <- function(charZip) {
  
  val <- as.numeric(rentalIndex[which(rentalIndex$zip == charZip),"rentalindex"])
  if (is.na(val)) {
    nearestZipWithData <- getNearestZip(charZip)
    val <- as.numeric(rentalIndex[which(rentalIndex$zip == nearestZipWithData),"rentalindex"])
  }
  return(val)
}

getCityFromZip <- function(charZip) {
  city <- zipcode %>%
    filter(zip == charZip) %>%
    select(city) %>%
    as.character()
  return(city)
}

# get the coordinates of the zipcode
getCoordFromZip <- function(charZip) {
  coord <- zipcode %>%
    filter(zip == charZip) %>%
    select(longitude, latitude) %>%
    as.matrix()
  return(coord)
}

# get the zipcode closest to the query zipcode for which we have 
# data populated in the Rental Price Index
getNearestZip <- function(charZip) {
  zipWithData <- rentalIndex %>%
    select(zip)
  
  query_city <- getCityFromZip(charZip)
  
  if (is.na(query_city)) {
    return("10026")
  }
  
  query_space <- zipcode %>%
    filter(city == query_city) 
  

  
  query_points <- query_space %>%
    select(longitude, latitude) %>%
    as.matrix()
  
  target_coord <- getCoordFromZip(charZip)
  
  if (is.na(target_coord)) {
    return("10026")
  }
  
  m <- distm(target_coord, query_points)
  i <- which.min(m)
  return(query_space$zip[i])
}


