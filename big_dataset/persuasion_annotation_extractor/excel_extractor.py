import pandas as pd
import openpyxl
# import warnings
#
# # def fxn():
# #     warnings.warn("Unknown extension is not supported and will be removed", UserWarning)
#
#
# with warnings.catch_warnings():
#     # warnings.simplefilter("ignore")
#     warnings.filterwarnings("ignore", category=UserWarning, module='_reader')
#     # fxn()

### there is an error message about "UserWarning: Unknown extension is not supported and will be removed"
### this is unimportant -- it refers to the data validation (drop-down menus) we created in google sheets. does not
### affect the code

xls = pd.ExcelFile('../../COSI140B/persuasion_annotation_extractor/full_dialog_annotations.xlsx')
data = pd.read_excel(xls, 'annotator_3_and_master')
print(data.head()) # first five rows of all columns

# create DataFrame using data, includes the sentences, annotator_1's tags, annotator_2's tags, adjudicator's tags
df = pd.DataFrame(data, columns=['Line_Num', 'Text', 'Annotation_1', 'Annotation_2', 'Annotation_3'])
df['Line_Num'] = df['Line_Num'].astype(int)
print(df.head()) # first five rows with selected columns

# Create variable TRUE if line has annotation_1
annotated_1 = ~df['Annotation_1'].isnull()

# Create variable TRUE if line has annotation_2
annotated_2 = ~df['Annotation_2'].isnull()

# Create variable TRUE if line has annotation_3
adjudicated = ~df['Annotation_3'].isnull()

# splits dataset to only include rows that have annotations = 'train'
non_nulls = df[annotated_1 & annotated_2][['Line_Num', 'Text', 'Annotation_1', 'Annotation_2', 'Annotation_3']]
print(non_nulls) # 791 annotated (unadjudicated and adjudicated) rows
# or
adjudicated_lines = df[adjudicated][['Line_Num', 'Text', 'Annotation_1', 'Annotation_2', 'Annotation_3']]
print(adjudicated_lines) # 175 annotated & adjudicated rows

# splits dataset to only include rows that do not have annotations yet
nulls = df[df['Annotation_1'].isnull()][['Line_Num', 'Text', 'Annotation_1', 'Annotation_2', 'Annotation_3']]
print(nulls) # 20141 un-annotated rows

# splits 'dev' set to separate 'test' files
