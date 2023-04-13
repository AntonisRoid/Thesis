library(corrplot)
library(RColorBrewer)
my_data <- read.csv('df_clean_plus.csv')

M<-cor(my_data[,c('temp_avg','sun_perc','precip_tot','pres_avg','wind_avg','cloud_avg','hum_avg')])

corrplot(M,type='upper',order='AOE',
         col= brewer.pal(n=7, name='RdYlBu'), method= 'ellipse')
         diag= FALSE)
