from model import MLModel

model = MLModel()
#
# model.upload_holidays(file_path='excel_files/feiertage-2020-nrw.xlsx',
#                       table_name='feiertage2020nrw')
# model.upload_holidays(file_path='excel_files/feiertage-2021-nrw.xlsx',
#                       table_name='feiertage2021nrw')
# model.upload_holidays(file_path='excel_files/feiertage-2022-nrw.xlsx',
#                       table_name='feiertage2022nrw')

# data = model._feature_extraction(table_name='stromlastdaten',
#                                  date_col='zeit',
#                                  holiday_tables=['feiertage2020nrw',
#                                                  'feiertage2021nrw',
#                                                  'feiertage2022nrw'])

preds = model.train_model()

sum_preds = model.export_predictions(predictions=preds)

sum_preds.to_csv('excel_files/predictions.csv', sep=',')
print('done')