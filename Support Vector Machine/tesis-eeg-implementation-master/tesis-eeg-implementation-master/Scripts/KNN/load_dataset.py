import pandas as pd

memory_files = ['Memoria2016310_1840.csv','Memoria2016310_1920.csv','Memoria2016422_1616.csv','Memoria2016422_1645.csv',
				'Memoria2016422_1712.csv','Memoria2016422_1749.csv', 'Memoria2016422_1823.csv','Memoria2016422_1850.csv'];
relax_files = ['Relajacion_wm_a12017914_2012.csv','Relajacion_wm_a22017914_2030.csv','Relajacion_wm_a32017914_2040.csv','Relajacion_wm_a42017914_213.csv',
				'Relajacion_wm_y12017914_2021.csv','Relajacion_wm_y22017914_2035.csv','Relajacion_wm_y32017914_2047.csv','Relajacion_wm_y42017914_2111.csv'];
relax_music_files = ['relajacion2016310_1852.csv','Relajacion2016310_1931.csv','Relajacion2016422_188.csv','Relajacion2016422_1628.csv',
					'relajacion2016422_1657.csv','Relajacion2016422_1723.csv','Relajacion2016422_1738.csv','Relajacion2016422_1839.csv'];

# Carga los archivos crudos, obtiene unicamente los datos de los electrodos
# y corta cada uno de los archivo un 10% al inicio y al final
def load_files(files,percent):
	df = pd.DataFrame();

	for file in files:		
		temp = pd.read_csv("../../DataSet/"+file,header=None)		
		temp = temp[[3,7,9,12,16]][1:].apply(pd.to_numeric)	
		cut_rows = int(len(temp.index) * percent);		
		temp = temp[:][cut_rows:len(temp.index) - cut_rows]			
		df = df.append(temp, ignore_index=True);		
	return df;
percent = input("Percent of the data to cut in both sides? %")
percent = int(percent) / 100;
memory_df = load_files(memory_files,percent);
relax_df = load_files(relax_files,percent);
relax_music_df = load_files(relax_music_files,percent);


# Guarda los archivos ya modificados en un csv
memory_df.to_csv('memory_dataset.csv');
relax_df.to_csv('relax_dataset.csv');
relax_music_df.to_csv('relax_music_dataset.csv');

print("Success!!")





