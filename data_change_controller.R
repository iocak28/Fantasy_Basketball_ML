rm(list = ls()); gc()

library(data.table)

old_path = 'C:/Users/iocak/OneDrive/Masaüstü/WI20/ECE 271B/Project/sample_data/player_data_yedek/'
new_path = 'C:/Users/iocak/OneDrive/Masaüstü/WI20/ECE 271B/Project/sample_data/player_data/'


files = list.files(old_path)
gate = 0

for (i in files){
  old = fread(paste0(old_path, i))
  new = fread(paste0(new_path, i))
  
  old[, V1 := NULL,]
  new[, 'Unnamed: 0' := NULL,]
  
  all.equal(old, new[, c(1:25)])
  
  old_met = old[, .(assists_mean = mean(assists), assists_sd = sd(assists)),]
  new_met = new[, .(assists_mean = mean(assists), assists_sd = sd(assists)),]
  
  if(all.equal(old, new[, c(1:25)]) == FALSE | all.equal(old_met, new_met) == FALSE){
    print(paste0(i, ' Problem'))
    gate = 1
  }
  
}
