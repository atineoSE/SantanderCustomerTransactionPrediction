# Only for coremltools version from PR at https://github.com/apple/coremltools/pull/293

import pandas as pd
import numpy as np
from coremltools.converters.xgboost import convert
from coremltools.models.utils import evaluate_regressor
import pickle

test = pd.read_csv("../data/train.csv")
test_xgb = (test.drop(['ID_code', 'target'], axis=1, inplace=False)).values
rename_dict = {'var_0':'f0', 'var_1':'f1', 'var_2':'f2', 'var_3':'f3', 'var_4':'f4', 'var_5':'f5', 'var_6':'f6', 'var_7':'f7', 'var_8':'f8', 'var_9':'f9', 'var_10':'f10', 'var_11':'f11', 'var_12':'f12', 'var_13':'f13', 'var_14':'f14', 'var_15':'f15', 'var_16':'f16', 'var_17':'f17', 'var_18':'f18', 'var_19':'f19', 'var_20':'f20', 'var_21':'f21', 'var_22':'f22', 'var_23':'f23', 'var_24':'f24', 'var_25':'f25', 'var_26':'f26', 'var_27':'f27', 'var_28':'f28', 'var_29':'f29', 'var_30':'f30', 'var_31':'f31', 'var_32':'f32', 'var_33':'f33', 'var_34':'f34', 'var_35':'f35', 'var_36':'f36', 'var_37':'f37', 'var_38':'f38', 'var_39':'f39', 'var_40':'f40', 'var_41':'f41', 'var_42':'f42', 'var_43':'f43', 'var_44':'f44', 'var_45':'f45', 'var_46':'f46', 'var_47':'f47', 'var_48':'f48', 'var_49':'f49', 'var_50':'f50', 'var_51':'f51', 'var_52':'f52', 'var_53':'f53', 'var_54':'f54', 'var_55':'f55', 'var_56':'f56', 'var_57':'f57', 'var_58':'f58', 'var_59':'f59', 'var_60':'f60', 'var_61':'f61', 'var_62':'f62', 'var_63':'f63', 'var_64':'f64', 'var_65':'f65', 'var_66':'f66', 'var_67':'f67', 'var_68':'f68', 'var_69':'f69', 'var_70':'f70', 'var_71':'f71', 'var_72':'f72', 'var_73':'f73', 'var_74':'f74', 'var_75':'f75', 'var_76':'f76', 'var_77':'f77', 'var_78':'f78', 'var_79':'f79', 'var_80':'f80', 'var_81':'f81', 'var_82':'f82', 'var_83':'f83', 'var_84':'f84', 'var_85':'f85', 'var_86':'f86', 'var_87':'f87', 'var_88':'f88', 'var_89':'f89', 'var_90':'f90', 'var_91':'f91', 'var_92':'f92', 'var_93':'f93', 'var_94':'f94', 'var_95':'f95', 'var_96':'f96', 'var_97':'f97', 'var_98':'f98', 'var_99':'f99', 'var_100':'f100', 'var_101':'f101', 'var_102':'f102', 'var_103':'f103', 'var_104':'f104', 'var_105':'f105', 'var_106':'f106', 'var_107':'f107', 'var_108':'f108', 'var_109':'f109', 'var_110':'f110', 'var_111':'f111', 'var_112':'f112', 'var_113':'f113', 'var_114':'f114', 'var_115':'f115', 'var_116':'f116', 'var_117':'f117', 'var_118':'f118', 'var_119':'f119', 'var_120':'f120', 'var_121':'f121', 'var_122':'f122', 'var_123':'f123', 'var_124':'f124', 'var_125':'f125', 'var_126':'f126', 'var_127':'f127', 'var_128':'f128', 'var_129':'f129', 'var_130':'f130', 'var_131':'f131', 'var_132':'f132', 'var_133':'f133', 'var_134':'f134', 'var_135':'f135', 'var_136':'f136', 'var_137':'f137', 'var_138':'f138', 'var_139':'f139', 'var_140':'f140', 'var_141':'f141', 'var_142':'f142', 'var_143':'f143', 'var_144':'f144', 'var_145':'f145', 'var_146':'f146', 'var_147':'f147', 'var_148':'f148', 'var_149':'f149', 'var_150':'f150', 'var_151':'f151', 'var_152':'f152', 'var_153':'f153', 'var_154':'f154', 'var_155':'f155', 'var_156':'f156', 'var_157':'f157', 'var_158':'f158', 'var_159':'f159', 'var_160':'f160', 'var_161':'f161', 'var_162':'f162', 'var_163':'f163', 'var_164':'f164', 'var_165':'f165', 'var_166':'f166', 'var_167':'f167', 'var_168':'f168', 'var_169':'f169', 'var_170':'f170', 'var_171':'f171', 'var_172':'f172', 'var_173':'f173', 'var_174':'f174', 'var_175':'f175', 'var_176':'f176', 'var_177':'f177', 'var_178':'f178', 'var_179':'f179', 'var_180':'f180', 'var_181':'f181', 'var_182':'f182', 'var_183':'f183', 'var_184':'f184', 'var_185':'f185', 'var_186':'f186', 'var_187':'f187', 'var_188':'f188', 'var_189':'f189', 'var_190':'f190', 'var_191':'f191', 'var_192':'f192', 'var_193':'f193', 'var_194':'f194', 'var_195':'f195', 'var_196':'f196', 'var_197':'f197', 'var_198':'f198', 'var_199':'f199'}
test_coreml = test.rename(columns=rename_dict, inplace=False)
print(test_coreml.describe())

models = ["xgb_fold{0}.dat".format(i) for i in range(1,6)]
for idx, m in enumerate(models):
    print("Converting {0}".format(m))
    xgb_model = pickle.load(open("./Models/"+m, "rb"))
    predictions_xgb = xgb_model.predict(test_xgb)

    test_coreml["prediction"] = pd.Series(predictions_xgb)

    coreml_model = convert(xgb_model, mode="classifier")
    metrics = evaluate_regressor(coreml_model, test_coreml, target="target", verbose=False)
    print("coreml prediction metrics")
    print(metrics)
    print("\n")
    coreml_model.save("./Models/XgbClassifier{0}.mlmodel".format(idx+1))
