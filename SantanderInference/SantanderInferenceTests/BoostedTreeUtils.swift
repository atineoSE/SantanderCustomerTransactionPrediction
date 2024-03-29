//
//  BoostedTreeUtils.swift
//  SantanderInferenceTests
//
//  Created by Adrian Tineo on 29.05.19.
//  Copyright © 2019 adriantineo.com. All rights reserved.
//

import Foundation
import CoreML

@testable import SantanderInference

func makeInputSantanderBoostedTree_100_000(row: String) -> SantanderBoostedTreeRegressor_100_000_itInput {
    
    let values = row.split(separator: ",")
    
    let var_0 = Double(values[2])!
    let var_1 = Double(values[3])!
    let var_2 = Double(values[4])!
    let var_3 = Double(values[5])!
    let var_4 = Double(values[6])!
    let var_5 = Double(values[7])!
    let var_6 = Double(values[8])!
    let var_7 = Double(values[9])!
    let var_8 = Double(values[10])!
    let var_9 = Double(values[11])!
    let var_10 = Double(values[12])!
    let var_11 = Double(values[13])!
    let var_12 = Double(values[14])!
    let var_13 = Double(values[15])!
    let var_14 = Double(values[16])!
    let var_15 = Double(values[17])!
    let var_16 = Double(values[18])!
    let var_17 = Double(values[19])!
    let var_18 = Double(values[20])!
    let var_19 = Double(values[21])!
    let var_20 = Double(values[22])!
    let var_21 = Double(values[23])!
    let var_22 = Double(values[24])!
    let var_23 = Double(values[25])!
    let var_24 = Double(values[26])!
    let var_25 = Double(values[27])!
    let var_26 = Double(values[28])!
    let var_27 = Double(values[29])!
    let var_28 = Double(values[30])!
    let var_29 = Double(values[31])!
    let var_30 = Double(values[32])!
    let var_31 = Double(values[33])!
    let var_32 = Double(values[34])!
    let var_33 = Double(values[35])!
    let var_34 = Double(values[36])!
    let var_35 = Double(values[37])!
    let var_36 = Double(values[38])!
    let var_37 = Double(values[39])!
    let var_38 = Double(values[40])!
    let var_39 = Double(values[41])!
    let var_40 = Double(values[42])!
    let var_41 = Double(values[43])!
    let var_42 = Double(values[44])!
    let var_43 = Double(values[45])!
    let var_44 = Double(values[46])!
    let var_45 = Double(values[47])!
    let var_46 = Double(values[48])!
    let var_47 = Double(values[49])!
    let var_48 = Double(values[50])!
    let var_49 = Double(values[51])!
    let var_50 = Double(values[52])!
    let var_51 = Double(values[53])!
    let var_52 = Double(values[54])!
    let var_53 = Double(values[55])!
    let var_54 = Double(values[56])!
    let var_55 = Double(values[57])!
    let var_56 = Double(values[58])!
    let var_57 = Double(values[59])!
    let var_58 = Double(values[60])!
    let var_59 = Double(values[61])!
    let var_60 = Double(values[62])!
    let var_61 = Double(values[63])!
    let var_62 = Double(values[64])!
    let var_63 = Double(values[65])!
    let var_64 = Double(values[66])!
    let var_65 = Double(values[67])!
    let var_66 = Double(values[68])!
    let var_67 = Double(values[69])!
    let var_68 = Double(values[70])!
    let var_69 = Double(values[71])!
    let var_70 = Double(values[72])!
    let var_71 = Double(values[73])!
    let var_72 = Double(values[74])!
    let var_73 = Double(values[75])!
    let var_74 = Double(values[76])!
    let var_75 = Double(values[77])!
    let var_76 = Double(values[78])!
    let var_77 = Double(values[79])!
    let var_78 = Double(values[80])!
    let var_79 = Double(values[81])!
    let var_80 = Double(values[82])!
    let var_81 = Double(values[83])!
    let var_82 = Double(values[84])!
    let var_83 = Double(values[85])!
    let var_84 = Double(values[86])!
    let var_85 = Double(values[87])!
    let var_86 = Double(values[88])!
    let var_87 = Double(values[89])!
    let var_88 = Double(values[90])!
    let var_89 = Double(values[91])!
    let var_90 = Double(values[92])!
    let var_91 = Double(values[93])!
    let var_92 = Double(values[94])!
    let var_93 = Double(values[95])!
    let var_94 = Double(values[96])!
    let var_95 = Double(values[97])!
    let var_96 = Double(values[98])!
    let var_97 = Double(values[99])!
    let var_98 = Double(values[100])!
    let var_99 = Double(values[101])!
    let var_100 = Double(values[102])!
    let var_101 = Double(values[103])!
    let var_102 = Double(values[104])!
    let var_103 = Double(values[105])!
    let var_104 = Double(values[106])!
    let var_105 = Double(values[107])!
    let var_106 = Double(values[108])!
    let var_107 = Double(values[109])!
    let var_108 = Double(values[110])!
    let var_109 = Double(values[111])!
    let var_110 = Double(values[112])!
    let var_111 = Double(values[113])!
    let var_112 = Double(values[114])!
    let var_113 = Double(values[115])!
    let var_114 = Double(values[116])!
    let var_115 = Double(values[117])!
    let var_116 = Double(values[118])!
    let var_117 = Double(values[119])!
    let var_118 = Double(values[120])!
    let var_119 = Double(values[121])!
    let var_120 = Double(values[122])!
    let var_121 = Double(values[123])!
    let var_122 = Double(values[124])!
    let var_123 = Double(values[125])!
    let var_124 = Double(values[126])!
    let var_125 = Double(values[127])!
    let var_126 = Double(values[128])!
    let var_127 = Double(values[129])!
    let var_128 = Double(values[130])!
    let var_129 = Double(values[131])!
    let var_130 = Double(values[132])!
    let var_131 = Double(values[133])!
    let var_132 = Double(values[134])!
    let var_133 = Double(values[135])!
    let var_134 = Double(values[136])!
    let var_135 = Double(values[137])!
    let var_136 = Double(values[138])!
    let var_137 = Double(values[139])!
    let var_138 = Double(values[140])!
    let var_139 = Double(values[141])!
    let var_140 = Double(values[142])!
    let var_141 = Double(values[143])!
    let var_142 = Double(values[144])!
    let var_143 = Double(values[145])!
    let var_144 = Double(values[146])!
    let var_145 = Double(values[147])!
    let var_146 = Double(values[148])!
    let var_147 = Double(values[149])!
    let var_148 = Double(values[150])!
    let var_149 = Double(values[151])!
    let var_150 = Double(values[152])!
    let var_151 = Double(values[153])!
    let var_152 = Double(values[154])!
    let var_153 = Double(values[155])!
    let var_154 = Double(values[156])!
    let var_155 = Double(values[157])!
    let var_156 = Double(values[158])!
    let var_157 = Double(values[159])!
    let var_158 = Double(values[160])!
    let var_159 = Double(values[161])!
    let var_160 = Double(values[162])!
    let var_161 = Double(values[163])!
    let var_162 = Double(values[164])!
    let var_163 = Double(values[165])!
    let var_164 = Double(values[166])!
    let var_165 = Double(values[167])!
    let var_166 = Double(values[168])!
    let var_167 = Double(values[169])!
    let var_168 = Double(values[170])!
    let var_169 = Double(values[171])!
    let var_170 = Double(values[172])!
    let var_171 = Double(values[173])!
    let var_172 = Double(values[174])!
    let var_173 = Double(values[175])!
    let var_174 = Double(values[176])!
    let var_175 = Double(values[177])!
    let var_176 = Double(values[178])!
    let var_177 = Double(values[179])!
    let var_178 = Double(values[180])!
    let var_179 = Double(values[181])!
    let var_180 = Double(values[182])!
    let var_181 = Double(values[183])!
    let var_182 = Double(values[184])!
    let var_183 = Double(values[185])!
    let var_184 = Double(values[186])!
    let var_185 = Double(values[187])!
    let var_186 = Double(values[188])!
    let var_187 = Double(values[189])!
    let var_188 = Double(values[190])!
    let var_189 = Double(values[191])!
    let var_190 = Double(values[192])!
    let var_191 = Double(values[193])!
    let var_192 = Double(values[194])!
    let var_193 = Double(values[195])!
    let var_194 = Double(values[196])!
    let var_195 = Double(values[197])!
    let var_196 = Double(values[198])!
    let var_197 = Double(values[199])!
    let var_198 = Double(values[200])!
    let var_199 = Double(values[201])!

    
        return SantanderBoostedTreeRegressor_100_000_itInput(var_0: var_0, var_1: var_1, var_2: var_2, var_3: var_3, var_4: var_4, var_5: var_5, var_6: var_6, var_7: var_7, var_8: var_8, var_9: var_9, var_10: var_10, var_11: var_11, var_12: var_12, var_13: var_13, var_14: var_14, var_15: var_15, var_16: var_16, var_17: var_17, var_18: var_18, var_19: var_19, var_20: var_20, var_21: var_21, var_22: var_22, var_23: var_23, var_24: var_24, var_25: var_25, var_26: var_26, var_27: var_27, var_28: var_28, var_29: var_29, var_30: var_30, var_31: var_31, var_32: var_32, var_33: var_33, var_34: var_34, var_35: var_35, var_36: var_36, var_37: var_37, var_38: var_38, var_39: var_39, var_40: var_40, var_41: var_41, var_42: var_42, var_43: var_43, var_44: var_44, var_45: var_45, var_46: var_46, var_47: var_47, var_48: var_48, var_49: var_49, var_50: var_50, var_51: var_51, var_52: var_52, var_53: var_53, var_54: var_54, var_55: var_55, var_56: var_56, var_57: var_57, var_58: var_58, var_59: var_59, var_60: var_60, var_61: var_61, var_62: var_62, var_63: var_63, var_64: var_64, var_65: var_65, var_66: var_66, var_67: var_67, var_68: var_68, var_69: var_69, var_70: var_70, var_71: var_71, var_72: var_72, var_73: var_73, var_74: var_74, var_75: var_75, var_76: var_76, var_77: var_77, var_78: var_78, var_79: var_79, var_80: var_80, var_81: var_81, var_82: var_82, var_83: var_83, var_84: var_84, var_85: var_85, var_86: var_86, var_87: var_87, var_88: var_88, var_89: var_89, var_90: var_90, var_91: var_91, var_92: var_92, var_93: var_93, var_94: var_94, var_95: var_95, var_96: var_96, var_97: var_97, var_98: var_98, var_99: var_99, var_100: var_100, var_101: var_101, var_102: var_102, var_103: var_103, var_104: var_104, var_105: var_105, var_106: var_106, var_107: var_107, var_108: var_108, var_109: var_109, var_110: var_110, var_111: var_111, var_112: var_112, var_113: var_113, var_114: var_114, var_115: var_115, var_116: var_116, var_117: var_117, var_118: var_118, var_119: var_119, var_120: var_120, var_121: var_121, var_122: var_122, var_123: var_123, var_124: var_124, var_125: var_125, var_126: var_126, var_127: var_127, var_128: var_128, var_129: var_129, var_130: var_130, var_131: var_131, var_132: var_132, var_133: var_133, var_134: var_134, var_135: var_135, var_136: var_136, var_137: var_137, var_138: var_138, var_139: var_139, var_140: var_140, var_141: var_141, var_142: var_142, var_143: var_143, var_144: var_144, var_145: var_145, var_146: var_146, var_147: var_147, var_148: var_148, var_149: var_149, var_150: var_150, var_151: var_151, var_152: var_152, var_153: var_153, var_154: var_154, var_155: var_155, var_156: var_156, var_157: var_157, var_158: var_158, var_159: var_159, var_160: var_160, var_161: var_161, var_162: var_162, var_163: var_163, var_164: var_164, var_165: var_165, var_166: var_166, var_167: var_167, var_168: var_168, var_169: var_169, var_170: var_170, var_171: var_171, var_172: var_172, var_173: var_173, var_174: var_174, var_175: var_175, var_176: var_176, var_177: var_177, var_178: var_178, var_179: var_179, var_180: var_180, var_181: var_181, var_182: var_182, var_183: var_183, var_184: var_184, var_185: var_185, var_186: var_186, var_187: var_187, var_188: var_188, var_189: var_189, var_190: var_190, var_191: var_191, var_192: var_192, var_193: var_193, var_194: var_194, var_195: var_195, var_196: var_196, var_197: var_197, var_198: var_198, var_199: var_199)
}

func makeInputSantanderBoostedTree_20_000(row: String) -> SantanderBoostedTreeRegressor_20_000_itInput {
    
    let values = row.split(separator: ",")
    
    let var_0 = Double(values[2])!
    let var_1 = Double(values[3])!
    let var_2 = Double(values[4])!
    let var_3 = Double(values[5])!
    let var_4 = Double(values[6])!
    let var_5 = Double(values[7])!
    let var_6 = Double(values[8])!
    let var_7 = Double(values[9])!
    let var_8 = Double(values[10])!
    let var_9 = Double(values[11])!
    let var_10 = Double(values[12])!
    let var_11 = Double(values[13])!
    let var_12 = Double(values[14])!
    let var_13 = Double(values[15])!
    let var_14 = Double(values[16])!
    let var_15 = Double(values[17])!
    let var_16 = Double(values[18])!
    let var_17 = Double(values[19])!
    let var_18 = Double(values[20])!
    let var_19 = Double(values[21])!
    let var_20 = Double(values[22])!
    let var_21 = Double(values[23])!
    let var_22 = Double(values[24])!
    let var_23 = Double(values[25])!
    let var_24 = Double(values[26])!
    let var_25 = Double(values[27])!
    let var_26 = Double(values[28])!
    let var_27 = Double(values[29])!
    let var_28 = Double(values[30])!
    let var_29 = Double(values[31])!
    let var_30 = Double(values[32])!
    let var_31 = Double(values[33])!
    let var_32 = Double(values[34])!
    let var_33 = Double(values[35])!
    let var_34 = Double(values[36])!
    let var_35 = Double(values[37])!
    let var_36 = Double(values[38])!
    let var_37 = Double(values[39])!
    let var_38 = Double(values[40])!
    let var_39 = Double(values[41])!
    let var_40 = Double(values[42])!
    let var_41 = Double(values[43])!
    let var_42 = Double(values[44])!
    let var_43 = Double(values[45])!
    let var_44 = Double(values[46])!
    let var_45 = Double(values[47])!
    let var_46 = Double(values[48])!
    let var_47 = Double(values[49])!
    let var_48 = Double(values[50])!
    let var_49 = Double(values[51])!
    let var_50 = Double(values[52])!
    let var_51 = Double(values[53])!
    let var_52 = Double(values[54])!
    let var_53 = Double(values[55])!
    let var_54 = Double(values[56])!
    let var_55 = Double(values[57])!
    let var_56 = Double(values[58])!
    let var_57 = Double(values[59])!
    let var_58 = Double(values[60])!
    let var_59 = Double(values[61])!
    let var_60 = Double(values[62])!
    let var_61 = Double(values[63])!
    let var_62 = Double(values[64])!
    let var_63 = Double(values[65])!
    let var_64 = Double(values[66])!
    let var_65 = Double(values[67])!
    let var_66 = Double(values[68])!
    let var_67 = Double(values[69])!
    let var_68 = Double(values[70])!
    let var_69 = Double(values[71])!
    let var_70 = Double(values[72])!
    let var_71 = Double(values[73])!
    let var_72 = Double(values[74])!
    let var_73 = Double(values[75])!
    let var_74 = Double(values[76])!
    let var_75 = Double(values[77])!
    let var_76 = Double(values[78])!
    let var_77 = Double(values[79])!
    let var_78 = Double(values[80])!
    let var_79 = Double(values[81])!
    let var_80 = Double(values[82])!
    let var_81 = Double(values[83])!
    let var_82 = Double(values[84])!
    let var_83 = Double(values[85])!
    let var_84 = Double(values[86])!
    let var_85 = Double(values[87])!
    let var_86 = Double(values[88])!
    let var_87 = Double(values[89])!
    let var_88 = Double(values[90])!
    let var_89 = Double(values[91])!
    let var_90 = Double(values[92])!
    let var_91 = Double(values[93])!
    let var_92 = Double(values[94])!
    let var_93 = Double(values[95])!
    let var_94 = Double(values[96])!
    let var_95 = Double(values[97])!
    let var_96 = Double(values[98])!
    let var_97 = Double(values[99])!
    let var_98 = Double(values[100])!
    let var_99 = Double(values[101])!
    let var_100 = Double(values[102])!
    let var_101 = Double(values[103])!
    let var_102 = Double(values[104])!
    let var_103 = Double(values[105])!
    let var_104 = Double(values[106])!
    let var_105 = Double(values[107])!
    let var_106 = Double(values[108])!
    let var_107 = Double(values[109])!
    let var_108 = Double(values[110])!
    let var_109 = Double(values[111])!
    let var_110 = Double(values[112])!
    let var_111 = Double(values[113])!
    let var_112 = Double(values[114])!
    let var_113 = Double(values[115])!
    let var_114 = Double(values[116])!
    let var_115 = Double(values[117])!
    let var_116 = Double(values[118])!
    let var_117 = Double(values[119])!
    let var_118 = Double(values[120])!
    let var_119 = Double(values[121])!
    let var_120 = Double(values[122])!
    let var_121 = Double(values[123])!
    let var_122 = Double(values[124])!
    let var_123 = Double(values[125])!
    let var_124 = Double(values[126])!
    let var_125 = Double(values[127])!
    let var_126 = Double(values[128])!
    let var_127 = Double(values[129])!
    let var_128 = Double(values[130])!
    let var_129 = Double(values[131])!
    let var_130 = Double(values[132])!
    let var_131 = Double(values[133])!
    let var_132 = Double(values[134])!
    let var_133 = Double(values[135])!
    let var_134 = Double(values[136])!
    let var_135 = Double(values[137])!
    let var_136 = Double(values[138])!
    let var_137 = Double(values[139])!
    let var_138 = Double(values[140])!
    let var_139 = Double(values[141])!
    let var_140 = Double(values[142])!
    let var_141 = Double(values[143])!
    let var_142 = Double(values[144])!
    let var_143 = Double(values[145])!
    let var_144 = Double(values[146])!
    let var_145 = Double(values[147])!
    let var_146 = Double(values[148])!
    let var_147 = Double(values[149])!
    let var_148 = Double(values[150])!
    let var_149 = Double(values[151])!
    let var_150 = Double(values[152])!
    let var_151 = Double(values[153])!
    let var_152 = Double(values[154])!
    let var_153 = Double(values[155])!
    let var_154 = Double(values[156])!
    let var_155 = Double(values[157])!
    let var_156 = Double(values[158])!
    let var_157 = Double(values[159])!
    let var_158 = Double(values[160])!
    let var_159 = Double(values[161])!
    let var_160 = Double(values[162])!
    let var_161 = Double(values[163])!
    let var_162 = Double(values[164])!
    let var_163 = Double(values[165])!
    let var_164 = Double(values[166])!
    let var_165 = Double(values[167])!
    let var_166 = Double(values[168])!
    let var_167 = Double(values[169])!
    let var_168 = Double(values[170])!
    let var_169 = Double(values[171])!
    let var_170 = Double(values[172])!
    let var_171 = Double(values[173])!
    let var_172 = Double(values[174])!
    let var_173 = Double(values[175])!
    let var_174 = Double(values[176])!
    let var_175 = Double(values[177])!
    let var_176 = Double(values[178])!
    let var_177 = Double(values[179])!
    let var_178 = Double(values[180])!
    let var_179 = Double(values[181])!
    let var_180 = Double(values[182])!
    let var_181 = Double(values[183])!
    let var_182 = Double(values[184])!
    let var_183 = Double(values[185])!
    let var_184 = Double(values[186])!
    let var_185 = Double(values[187])!
    let var_186 = Double(values[188])!
    let var_187 = Double(values[189])!
    let var_188 = Double(values[190])!
    let var_189 = Double(values[191])!
    let var_190 = Double(values[192])!
    let var_191 = Double(values[193])!
    let var_192 = Double(values[194])!
    let var_193 = Double(values[195])!
    let var_194 = Double(values[196])!
    let var_195 = Double(values[197])!
    let var_196 = Double(values[198])!
    let var_197 = Double(values[199])!
    let var_198 = Double(values[200])!
    let var_199 = Double(values[201])!
    
    
    return SantanderBoostedTreeRegressor_20_000_itInput(var_0: var_0, var_1: var_1, var_2: var_2, var_3: var_3, var_4: var_4, var_5: var_5, var_6: var_6, var_7: var_7, var_8: var_8, var_9: var_9, var_10: var_10, var_11: var_11, var_12: var_12, var_13: var_13, var_14: var_14, var_15: var_15, var_16: var_16, var_17: var_17, var_18: var_18, var_19: var_19, var_20: var_20, var_21: var_21, var_22: var_22, var_23: var_23, var_24: var_24, var_25: var_25, var_26: var_26, var_27: var_27, var_28: var_28, var_29: var_29, var_30: var_30, var_31: var_31, var_32: var_32, var_33: var_33, var_34: var_34, var_35: var_35, var_36: var_36, var_37: var_37, var_38: var_38, var_39: var_39, var_40: var_40, var_41: var_41, var_42: var_42, var_43: var_43, var_44: var_44, var_45: var_45, var_46: var_46, var_47: var_47, var_48: var_48, var_49: var_49, var_50: var_50, var_51: var_51, var_52: var_52, var_53: var_53, var_54: var_54, var_55: var_55, var_56: var_56, var_57: var_57, var_58: var_58, var_59: var_59, var_60: var_60, var_61: var_61, var_62: var_62, var_63: var_63, var_64: var_64, var_65: var_65, var_66: var_66, var_67: var_67, var_68: var_68, var_69: var_69, var_70: var_70, var_71: var_71, var_72: var_72, var_73: var_73, var_74: var_74, var_75: var_75, var_76: var_76, var_77: var_77, var_78: var_78, var_79: var_79, var_80: var_80, var_81: var_81, var_82: var_82, var_83: var_83, var_84: var_84, var_85: var_85, var_86: var_86, var_87: var_87, var_88: var_88, var_89: var_89, var_90: var_90, var_91: var_91, var_92: var_92, var_93: var_93, var_94: var_94, var_95: var_95, var_96: var_96, var_97: var_97, var_98: var_98, var_99: var_99, var_100: var_100, var_101: var_101, var_102: var_102, var_103: var_103, var_104: var_104, var_105: var_105, var_106: var_106, var_107: var_107, var_108: var_108, var_109: var_109, var_110: var_110, var_111: var_111, var_112: var_112, var_113: var_113, var_114: var_114, var_115: var_115, var_116: var_116, var_117: var_117, var_118: var_118, var_119: var_119, var_120: var_120, var_121: var_121, var_122: var_122, var_123: var_123, var_124: var_124, var_125: var_125, var_126: var_126, var_127: var_127, var_128: var_128, var_129: var_129, var_130: var_130, var_131: var_131, var_132: var_132, var_133: var_133, var_134: var_134, var_135: var_135, var_136: var_136, var_137: var_137, var_138: var_138, var_139: var_139, var_140: var_140, var_141: var_141, var_142: var_142, var_143: var_143, var_144: var_144, var_145: var_145, var_146: var_146, var_147: var_147, var_148: var_148, var_149: var_149, var_150: var_150, var_151: var_151, var_152: var_152, var_153: var_153, var_154: var_154, var_155: var_155, var_156: var_156, var_157: var_157, var_158: var_158, var_159: var_159, var_160: var_160, var_161: var_161, var_162: var_162, var_163: var_163, var_164: var_164, var_165: var_165, var_166: var_166, var_167: var_167, var_168: var_168, var_169: var_169, var_170: var_170, var_171: var_171, var_172: var_172, var_173: var_173, var_174: var_174, var_175: var_175, var_176: var_176, var_177: var_177, var_178: var_178, var_179: var_179, var_180: var_180, var_181: var_181, var_182: var_182, var_183: var_183, var_184: var_184, var_185: var_185, var_186: var_186, var_187: var_187, var_188: var_188, var_189: var_189, var_190: var_190, var_191: var_191, var_192: var_192, var_193: var_193, var_194: var_194, var_195: var_195, var_196: var_196, var_197: var_197, var_198: var_198, var_199: var_199)
}

func makeInputSantanderBoostedTree_2_500(row: String) -> SantanderBoostedTreeRegressor_2_500_itInput {
    
    let values = row.split(separator: ",")
    
    let var_0 = Double(values[2])!
    let var_1 = Double(values[3])!
    let var_2 = Double(values[4])!
    let var_3 = Double(values[5])!
    let var_4 = Double(values[6])!
    let var_5 = Double(values[7])!
    let var_6 = Double(values[8])!
    let var_7 = Double(values[9])!
    let var_8 = Double(values[10])!
    let var_9 = Double(values[11])!
    let var_10 = Double(values[12])!
    let var_11 = Double(values[13])!
    let var_12 = Double(values[14])!
    let var_13 = Double(values[15])!
    let var_14 = Double(values[16])!
    let var_15 = Double(values[17])!
    let var_16 = Double(values[18])!
    let var_17 = Double(values[19])!
    let var_18 = Double(values[20])!
    let var_19 = Double(values[21])!
    let var_20 = Double(values[22])!
    let var_21 = Double(values[23])!
    let var_22 = Double(values[24])!
    let var_23 = Double(values[25])!
    let var_24 = Double(values[26])!
    let var_25 = Double(values[27])!
    let var_26 = Double(values[28])!
    let var_27 = Double(values[29])!
    let var_28 = Double(values[30])!
    let var_29 = Double(values[31])!
    let var_30 = Double(values[32])!
    let var_31 = Double(values[33])!
    let var_32 = Double(values[34])!
    let var_33 = Double(values[35])!
    let var_34 = Double(values[36])!
    let var_35 = Double(values[37])!
    let var_36 = Double(values[38])!
    let var_37 = Double(values[39])!
    let var_38 = Double(values[40])!
    let var_39 = Double(values[41])!
    let var_40 = Double(values[42])!
    let var_41 = Double(values[43])!
    let var_42 = Double(values[44])!
    let var_43 = Double(values[45])!
    let var_44 = Double(values[46])!
    let var_45 = Double(values[47])!
    let var_46 = Double(values[48])!
    let var_47 = Double(values[49])!
    let var_48 = Double(values[50])!
    let var_49 = Double(values[51])!
    let var_50 = Double(values[52])!
    let var_51 = Double(values[53])!
    let var_52 = Double(values[54])!
    let var_53 = Double(values[55])!
    let var_54 = Double(values[56])!
    let var_55 = Double(values[57])!
    let var_56 = Double(values[58])!
    let var_57 = Double(values[59])!
    let var_58 = Double(values[60])!
    let var_59 = Double(values[61])!
    let var_60 = Double(values[62])!
    let var_61 = Double(values[63])!
    let var_62 = Double(values[64])!
    let var_63 = Double(values[65])!
    let var_64 = Double(values[66])!
    let var_65 = Double(values[67])!
    let var_66 = Double(values[68])!
    let var_67 = Double(values[69])!
    let var_68 = Double(values[70])!
    let var_69 = Double(values[71])!
    let var_70 = Double(values[72])!
    let var_71 = Double(values[73])!
    let var_72 = Double(values[74])!
    let var_73 = Double(values[75])!
    let var_74 = Double(values[76])!
    let var_75 = Double(values[77])!
    let var_76 = Double(values[78])!
    let var_77 = Double(values[79])!
    let var_78 = Double(values[80])!
    let var_79 = Double(values[81])!
    let var_80 = Double(values[82])!
    let var_81 = Double(values[83])!
    let var_82 = Double(values[84])!
    let var_83 = Double(values[85])!
    let var_84 = Double(values[86])!
    let var_85 = Double(values[87])!
    let var_86 = Double(values[88])!
    let var_87 = Double(values[89])!
    let var_88 = Double(values[90])!
    let var_89 = Double(values[91])!
    let var_90 = Double(values[92])!
    let var_91 = Double(values[93])!
    let var_92 = Double(values[94])!
    let var_93 = Double(values[95])!
    let var_94 = Double(values[96])!
    let var_95 = Double(values[97])!
    let var_96 = Double(values[98])!
    let var_97 = Double(values[99])!
    let var_98 = Double(values[100])!
    let var_99 = Double(values[101])!
    let var_100 = Double(values[102])!
    let var_101 = Double(values[103])!
    let var_102 = Double(values[104])!
    let var_103 = Double(values[105])!
    let var_104 = Double(values[106])!
    let var_105 = Double(values[107])!
    let var_106 = Double(values[108])!
    let var_107 = Double(values[109])!
    let var_108 = Double(values[110])!
    let var_109 = Double(values[111])!
    let var_110 = Double(values[112])!
    let var_111 = Double(values[113])!
    let var_112 = Double(values[114])!
    let var_113 = Double(values[115])!
    let var_114 = Double(values[116])!
    let var_115 = Double(values[117])!
    let var_116 = Double(values[118])!
    let var_117 = Double(values[119])!
    let var_118 = Double(values[120])!
    let var_119 = Double(values[121])!
    let var_120 = Double(values[122])!
    let var_121 = Double(values[123])!
    let var_122 = Double(values[124])!
    let var_123 = Double(values[125])!
    let var_124 = Double(values[126])!
    let var_125 = Double(values[127])!
    let var_126 = Double(values[128])!
    let var_127 = Double(values[129])!
    let var_128 = Double(values[130])!
    let var_129 = Double(values[131])!
    let var_130 = Double(values[132])!
    let var_131 = Double(values[133])!
    let var_132 = Double(values[134])!
    let var_133 = Double(values[135])!
    let var_134 = Double(values[136])!
    let var_135 = Double(values[137])!
    let var_136 = Double(values[138])!
    let var_137 = Double(values[139])!
    let var_138 = Double(values[140])!
    let var_139 = Double(values[141])!
    let var_140 = Double(values[142])!
    let var_141 = Double(values[143])!
    let var_142 = Double(values[144])!
    let var_143 = Double(values[145])!
    let var_144 = Double(values[146])!
    let var_145 = Double(values[147])!
    let var_146 = Double(values[148])!
    let var_147 = Double(values[149])!
    let var_148 = Double(values[150])!
    let var_149 = Double(values[151])!
    let var_150 = Double(values[152])!
    let var_151 = Double(values[153])!
    let var_152 = Double(values[154])!
    let var_153 = Double(values[155])!
    let var_154 = Double(values[156])!
    let var_155 = Double(values[157])!
    let var_156 = Double(values[158])!
    let var_157 = Double(values[159])!
    let var_158 = Double(values[160])!
    let var_159 = Double(values[161])!
    let var_160 = Double(values[162])!
    let var_161 = Double(values[163])!
    let var_162 = Double(values[164])!
    let var_163 = Double(values[165])!
    let var_164 = Double(values[166])!
    let var_165 = Double(values[167])!
    let var_166 = Double(values[168])!
    let var_167 = Double(values[169])!
    let var_168 = Double(values[170])!
    let var_169 = Double(values[171])!
    let var_170 = Double(values[172])!
    let var_171 = Double(values[173])!
    let var_172 = Double(values[174])!
    let var_173 = Double(values[175])!
    let var_174 = Double(values[176])!
    let var_175 = Double(values[177])!
    let var_176 = Double(values[178])!
    let var_177 = Double(values[179])!
    let var_178 = Double(values[180])!
    let var_179 = Double(values[181])!
    let var_180 = Double(values[182])!
    let var_181 = Double(values[183])!
    let var_182 = Double(values[184])!
    let var_183 = Double(values[185])!
    let var_184 = Double(values[186])!
    let var_185 = Double(values[187])!
    let var_186 = Double(values[188])!
    let var_187 = Double(values[189])!
    let var_188 = Double(values[190])!
    let var_189 = Double(values[191])!
    let var_190 = Double(values[192])!
    let var_191 = Double(values[193])!
    let var_192 = Double(values[194])!
    let var_193 = Double(values[195])!
    let var_194 = Double(values[196])!
    let var_195 = Double(values[197])!
    let var_196 = Double(values[198])!
    let var_197 = Double(values[199])!
    let var_198 = Double(values[200])!
    let var_199 = Double(values[201])!
    
    
    return SantanderBoostedTreeRegressor_2_500_itInput(var_0: var_0, var_1: var_1, var_2: var_2, var_3: var_3, var_4: var_4, var_5: var_5, var_6: var_6, var_7: var_7, var_8: var_8, var_9: var_9, var_10: var_10, var_11: var_11, var_12: var_12, var_13: var_13, var_14: var_14, var_15: var_15, var_16: var_16, var_17: var_17, var_18: var_18, var_19: var_19, var_20: var_20, var_21: var_21, var_22: var_22, var_23: var_23, var_24: var_24, var_25: var_25, var_26: var_26, var_27: var_27, var_28: var_28, var_29: var_29, var_30: var_30, var_31: var_31, var_32: var_32, var_33: var_33, var_34: var_34, var_35: var_35, var_36: var_36, var_37: var_37, var_38: var_38, var_39: var_39, var_40: var_40, var_41: var_41, var_42: var_42, var_43: var_43, var_44: var_44, var_45: var_45, var_46: var_46, var_47: var_47, var_48: var_48, var_49: var_49, var_50: var_50, var_51: var_51, var_52: var_52, var_53: var_53, var_54: var_54, var_55: var_55, var_56: var_56, var_57: var_57, var_58: var_58, var_59: var_59, var_60: var_60, var_61: var_61, var_62: var_62, var_63: var_63, var_64: var_64, var_65: var_65, var_66: var_66, var_67: var_67, var_68: var_68, var_69: var_69, var_70: var_70, var_71: var_71, var_72: var_72, var_73: var_73, var_74: var_74, var_75: var_75, var_76: var_76, var_77: var_77, var_78: var_78, var_79: var_79, var_80: var_80, var_81: var_81, var_82: var_82, var_83: var_83, var_84: var_84, var_85: var_85, var_86: var_86, var_87: var_87, var_88: var_88, var_89: var_89, var_90: var_90, var_91: var_91, var_92: var_92, var_93: var_93, var_94: var_94, var_95: var_95, var_96: var_96, var_97: var_97, var_98: var_98, var_99: var_99, var_100: var_100, var_101: var_101, var_102: var_102, var_103: var_103, var_104: var_104, var_105: var_105, var_106: var_106, var_107: var_107, var_108: var_108, var_109: var_109, var_110: var_110, var_111: var_111, var_112: var_112, var_113: var_113, var_114: var_114, var_115: var_115, var_116: var_116, var_117: var_117, var_118: var_118, var_119: var_119, var_120: var_120, var_121: var_121, var_122: var_122, var_123: var_123, var_124: var_124, var_125: var_125, var_126: var_126, var_127: var_127, var_128: var_128, var_129: var_129, var_130: var_130, var_131: var_131, var_132: var_132, var_133: var_133, var_134: var_134, var_135: var_135, var_136: var_136, var_137: var_137, var_138: var_138, var_139: var_139, var_140: var_140, var_141: var_141, var_142: var_142, var_143: var_143, var_144: var_144, var_145: var_145, var_146: var_146, var_147: var_147, var_148: var_148, var_149: var_149, var_150: var_150, var_151: var_151, var_152: var_152, var_153: var_153, var_154: var_154, var_155: var_155, var_156: var_156, var_157: var_157, var_158: var_158, var_159: var_159, var_160: var_160, var_161: var_161, var_162: var_162, var_163: var_163, var_164: var_164, var_165: var_165, var_166: var_166, var_167: var_167, var_168: var_168, var_169: var_169, var_170: var_170, var_171: var_171, var_172: var_172, var_173: var_173, var_174: var_174, var_175: var_175, var_176: var_176, var_177: var_177, var_178: var_178, var_179: var_179, var_180: var_180, var_181: var_181, var_182: var_182, var_183: var_183, var_184: var_184, var_185: var_185, var_186: var_186, var_187: var_187, var_188: var_188, var_189: var_189, var_190: var_190, var_191: var_191, var_192: var_192, var_193: var_193, var_194: var_194, var_195: var_195, var_196: var_196, var_197: var_197, var_198: var_198, var_199: var_199)
}
