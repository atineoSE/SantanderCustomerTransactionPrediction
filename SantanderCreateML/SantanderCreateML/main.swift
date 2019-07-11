//
//  main.swift
//  SantanderCreateML
//
//  Created by Adrian Tineo on 24.05.19.
//  Copyright © 2019 adriantineo.com. All rights reserved.
//

import Foundation
import CreateML

// MARK: Regressable
protocol MLRegressable {
    func predictions(from data: MLDataTable) throws -> MLUntypedColumn
    var targetColumn: String { get }
    func write(to fileURL: URL, metadata: MLModelMetadata?) throws
    func evaluation(on labeledData: MLDataTable) -> MLRegressorMetrics
}
extension MLRegressor: MLRegressable {}
extension MLLinearRegressor: MLRegressable {}
extension MLDecisionTreeRegressor: MLRegressable {}
extension MLBoostedTreeRegressor: MLRegressable {}
extension MLRandomForestRegressor: MLRegressable {}

// MARK: Evaluate methods

func showSamplePredictions(model: MLRegressable, entries: MLDataTable, name: String)  {
    var predictionTable = entries
    let predictions = try? model.predictions(from: entries)
    print("SAMPLE PREDICTIONS FOR MODEL \(name):")
    if let predictions = predictions {
        predictionTable.addColumn(predictions, named: "predictions")
        print(predictionTable)
    } else {
        print("ERROR FOR PREDICTIONS")
    }
}

//func evaluateXgbRegressor1(model: XgbRegressor1, input: MLDataTable) {
//    let rows = input.rows
//    let inputs = rows.map { makeInputXgbRegressor1(row: $0) }
//    let predictionsArray = (try! model.predictions(inputs: inputs)).map { $0.target }
//    let predictionsColumn = MLDataColumn(predictionsArray)
//    let expectedValuesColumn = input["target"].map { Double($0.intValue!) }
//    let diffColumn = (expectedValuesColumn - predictionsColumn).map { abs($0) }
//
//    print("Mean error: \(diffColumn.mean()!); Max error: \(diffColumn.max()!); Std error: \(diffColumn.std()!)")
//}
//
//func evaluateXgbRegressor2(model: XgbRegressor2, input: MLDataTable) {
//    let rows = input.rows
//    let inputs = rows.map { makeInputXgbRegressor2(row: $0) }
//    let predictionsArray = (try! model.predictions(inputs: inputs)).map { $0.target }
//    let predictionsColumn = MLDataColumn(predictionsArray)
//    let expectedValuesColumn = input["target"].map { Double($0.intValue!) }
//    let diffColumn = (expectedValuesColumn - predictionsColumn).map { abs($0) }
//
//    print("Mean error: \(diffColumn.mean()!); Max error: \(diffColumn.max()!); Std error: \(diffColumn.std()!)")
//}
//
//func evaluateXgbRegressor3(model: XgbRegressor3, input: MLDataTable) {
//    let rows = input.rows
//    let inputs = rows.map { makeInputXgbRegressor3(row: $0) }
//    let predictionsArray = (try! model.predictions(inputs: inputs)).map { $0.target }
//    let predictionsColumn = MLDataColumn(predictionsArray)
//    let expectedValuesColumn = input["target"].map { Double($0.intValue!) }
//    let diffColumn = (expectedValuesColumn - predictionsColumn).map { abs($0) }
//
//    print("Mean error: \(diffColumn.mean()!); Max error: \(diffColumn.max()!); Std error: \(diffColumn.std()!)")
//}

//func evaluateXgbModels(xgb1: XgbRegressor1, xgb2: XgbRegressor2, xgb3: XgbRegressor3, input: MLDataTable) {
//    let rows = input.rows
//    let inputs1 = rows.map { makeInputXgbRegressor1(row: $0) }
//    let inputs2 = rows.map { makeInputXgbRegressor2(row: $0) }
//    let inputs3 = rows.map { makeInputXgbRegressor3(row: $0) }
//    let predictionsArray1 = (try! xgb1.predictions(inputs: inputs1)).map { $0.target }
//    let predictionsColumn1 = MLDataColumn(predictionsArray1)
//    let predictionsArray2 = (try! xgb2.predictions(inputs: inputs2)).map { $0.target }
//    let predictionsColumn2 = MLDataColumn(predictionsArray2)
//    let predictionsArray3 = (try! xgb3.predictions(inputs: inputs3)).map { $0.target }
//    let predictionsColumn3 = MLDataColumn(predictionsArray3)
//
//    let expectedValuesColumn = input["target"].map { Double($0.intValue!) }
//    let averagePredictions = (predictionsColumn1 + predictionsColumn2 + predictionsColumn3 ) / 3.0
//    print(averagePredictions)
//    let diffColumn = (expectedValuesColumn - averagePredictions).map { abs($0) }
//
//    print("Mean error: \(diffColumn.mean()!); Max error: \(diffColumn.max()!); Std error: \(diffColumn.std()!)")
//}

func evaluateXgbModels(xgb1: XgbClassifier1, xgb2: XgbClassifier2, xgb3: XgbClassifier3, xgb4: XgbClassifier4, xgb5: XgbClassifier5, input: MLDataTable) {
    let rows = input.rows
    let inputs1 = rows.map { makeInputXgbClassifier1(row: $0) }
    let inputs2 = rows.map { makeInputXgbClassifier2(row: $0) }
    let inputs3 = rows.map { makeInputXgbClassifier3(row: $0) }
    let inputs4 = rows.map { makeInputXgbClassifier4(row: $0) }
    let inputs5 = rows.map { makeInputXgbClassifier5(row: $0) }
    
    let predictionsArray1 = (try! xgb1.predictions(inputs: inputs1)).map { $0.classProbability[1]! }
    // $0.classProbability looks like [1: 0.009114004223548541, 0: 0.9908859957764514]
    // we take $0.classProbability[1]!, i.e. the probability of belonging to class 1
    let predictionsColumn1 = MLDataColumn(predictionsArray1)
    let predictionsArray2 = (try! xgb2.predictions(inputs: inputs2)).map { $0.classProbability[1]! }
    let predictionsColumn2 = MLDataColumn(predictionsArray2)
    let predictionsArray3 = (try! xgb3.predictions(inputs: inputs3)).map { $0.classProbability[1]! }
    let predictionsColumn3 = MLDataColumn(predictionsArray3)
    let predictionsArray4 = (try! xgb4.predictions(inputs: inputs4)).map { $0.classProbability[1]! }
    let predictionsColumn4 = MLDataColumn(predictionsArray4)
    let predictionsArray5 = (try! xgb5.predictions(inputs: inputs5)).map { $0.classProbability[1]! }
    let predictionsColumn5 = MLDataColumn(predictionsArray5)
    
    let expectedValuesColumn = input["target"].map { Double($0.intValue!) }
    
    let averagePredictions = (predictionsColumn1 + predictionsColumn2 + predictionsColumn3 + predictionsColumn4 + predictionsColumn5) / 5.0

    //print(averagePredictions)
    let diffColumn = (expectedValuesColumn - averagePredictions).map { abs($0) }

    print("Mean error: \(diffColumn.mean()!); Max error: \(diffColumn.max()!); Std error: \(diffColumn.std()!)")
}

func evaluateBoostedTree_100_000(model: SantanderBoostedTreeRegressor_100_000_it, input: MLDataTable) {
    let rows = input.rows
    let inputs = rows.map { makeInputSantanderBoostedTree_100_000(row: $0) }
    let predictionsArray = (try! model.predictions(inputs: inputs)).map { $0.target }
    let predictionsColumn = MLDataColumn(predictionsArray)
    let expectedValuesColumn = input["target"].map { Double($0.intValue!) }
    let diffColumn = (expectedValuesColumn - predictionsColumn).map { abs($0) }

    print("Mean error: \(diffColumn.mean()!); Max error: \(diffColumn.max()!); Std error: \(diffColumn.std()!)")
}

func evaluateBoostedTree_20_000(model: SantanderBoostedTreeRegressor_20_000_it, input: MLDataTable) {
    let rows = input.rows
    let inputs = rows.map { makeInputSantanderBoostedTree_20_000(row: $0) }
    let predictionsArray = (try! model.predictions(inputs: inputs)).map { $0.target }
    let predictionsColumn = MLDataColumn(predictionsArray)
    let expectedValuesColumn = input["target"].map { Double($0.intValue!) }
    let diffColumn = (expectedValuesColumn - predictionsColumn).map { abs($0) }
    
    print("Mean error: \(diffColumn.mean()!); Max error: \(diffColumn.max()!); Std error: \(diffColumn.std()!)")
}

func evaluateBoostedTree_2_500(model: SantanderBoostedTreeRegressor_2_500_it, input: MLDataTable) {
    let rows = input.rows
    let inputs = rows.map { makeInputSantanderBoostedTree_2_500(row: $0) }
    let predictionsArray = (try! model.predictions(inputs: inputs)).map { $0.target }
    let predictionsColumn = MLDataColumn(predictionsArray)
    let expectedValuesColumn = input["target"].map { Double($0.intValue!) }
    let diffColumn = (expectedValuesColumn - predictionsColumn).map { abs($0) }
    
    print("Mean error: \(diffColumn.mean()!); Max error: \(diffColumn.max()!); Std error: \(diffColumn.std()!)")
}

// MARK: Training methods
func createLinearModel(trainingData: MLDataTable, targetColumn: String, featureColumns: [String]) -> MLLinearRegressor? {
    let parameters = MLLinearRegressor.ModelParameters(validationData: nil,
                                                       maxIterations: 100,
                                                       l1Penalty: 0.0,
                                                       l2Penalty: 0.01,
                                                       stepSize: 0.02,
                                                       convergenceThreshold: 0.001,
                                                       featureRescaling: true)
    print("LINEAR REGRESSOR FOR \"\(targetColumn)\":")
    let regressor =  try? MLLinearRegressor(trainingData: trainingData,
                                            targetColumn: targetColumn,
                                            featureColumns: featureColumns,
                                            parameters: parameters)
    print(regressor?.description ?? "No model created")
    return regressor
}

func createDecisionTreeModel(trainingData: MLDataTable, targetColumn: String, featureColumns: [String]) -> MLDecisionTreeRegressor? {
    let parameters = MLDecisionTreeRegressor.ModelParameters(validationData: nil,
                                                             maxDepth: 12,
                                                             minLossReduction: 0.0,
                                                             minChildWeight: 0.1,
                                                             randomSeed: 42)
    print("DECISION TREE REGRESSOR FOR \"\(targetColumn)\":")
    let regressor = try? MLDecisionTreeRegressor(trainingData: trainingData,
                                                 targetColumn: targetColumn,
                                                 featureColumns: featureColumns,
                                                 parameters: parameters)
    
    print(regressor?.description ?? "No model created")
    return regressor
}

func createBoostedTreeModel(trainingData: MLDataTable, validationData: MLDataTable, targetColumn: String, featureColumns: [String]) -> MLBoostedTreeRegressor? {
    let parameters = MLBoostedTreeRegressor.ModelParameters(validationData: validationData,
                                                            maxDepth: 6,
                                                            maxIterations: 2500,
                                                            minLossReduction: 0.0,
                                                            minChildWeight: 0.1,
                                                            randomSeed: 42,
                                                            stepSize: 0.02,
                                                            earlyStoppingRounds: nil,
                                                            rowSubsample: 0.3,
                                                            columnSubsample: 0.3)
    print("BOOSTED TREE REGRESSOR FOR \"\(targetColumn)\":")
    let regressor = try? MLBoostedTreeRegressor(trainingData: trainingData,
                                                targetColumn: targetColumn,
                                                featureColumns: featureColumns,
                                                parameters: parameters)
    
    print(regressor?.description ?? "No model created")
    return regressor
}

func createRandomForestModel(trainingData: MLDataTable, targetColumn: String, featureColumns: [String]) -> MLRandomForestRegressor? {
    let parameters = MLRandomForestRegressor.ModelParameters(validationData: nil,
                                                             maxDepth: 6,
                                                             maxIterations: 10000,
                                                             minLossReduction: 0.0,
                                                             minChildWeight: 0.1,
                                                             randomSeed: 42,
                                                             rowSubsample: 0.3,
                                                             columnSubsample: 0.3)
    print("RANDOM FOREST REGRESSOR FOR \"\(targetColumn)\":")
    let regressor = try? MLRandomForestRegressor(trainingData: trainingData,
                                                 targetColumn: targetColumn,
                                                 featureColumns: featureColumns,
                                                 parameters: parameters)
    
    print(regressor?.description ?? "No model created")
    return regressor
}

func createModel(trainingData: MLDataTable, targetColumn: String, featureColumns: [String]) -> MLRegressor? {
    print("REGRESSOR FOR \"\(targetColumn)\":")
    let regressor =  try? MLRegressor(trainingData: trainingData,
                                      targetColumn: targetColumn,
                                      featureColumns: featureColumns)
    print(regressor?.description ?? "no model created")
    return regressor
}

func dump(table: MLDataTable, comment: String) {
    print(comment)
    print(table.description)
    print("SIZE: \(table.size)")
    if let error = table.error {
        print("\(comment) DATA ERROR: \(error)")
    }
}

// MARK: main
let start = CFAbsoluteTimeGetCurrent()

let imac = false
let prefixRows: Int? = nil
let basePath: String
if imac {
    basePath = "/Volumes/MEDIA&DEV/dev/ML_data/SantanderCustomerTransactionPrediction/"
} else {
    basePath = "/Users/adriantineo/dev/ML_data/SantanderCustomerTransactionPrediction/"
}
let train = true
let evaluate = false

// MARK: train
if train {
    // 1. Import data
    let trainingSetPath = URL(fileURLWithPath: basePath + "SantanderCreateML/SantanderCreateML/Data/train_pre.csv")
    let validationSetPath = URL(fileURLWithPath: basePath + "SantanderCreateML/SantanderCreateML/Data/validation_pre.csv")
   
    var parsingOptions = MLDataTable.ParsingOptions()
    parsingOptions.delimiter = ","
    parsingOptions.lineTerminator = "\n"
   
    // random split is not appropriate because it does not consider keeping the target classes balanced
    // let (trainingTable, testTable) = inputDataTable.randomSplit(by: 0.8)
    
    let trainingTable = try MLDataTable(contentsOf: trainingSetPath, options: parsingOptions)
    dump(table: trainingTable, comment: "TRAINING TABLE")
    let validationTable = try MLDataTable(contentsOf: validationSetPath, options: parsingOptions)
    dump(table: trainingTable, comment: "VALIDATION TABLE")
    
    // 2. Create ML models
    let trainingTablePrefix = (prefixRows != nil) ? trainingTable.prefix(prefixRows!) : trainingTable
    let featureColumns = (0..<200).map { "var_" + "\($0)" }
    //let linearModel = createLinearModel(trainingData: trainingTablePrefix,
    //                        targetColumn: "target",
    //                        featureColumns: featureColumns)!
    //
    //let decisionTreeModel = createDecisionTreeModel(trainingData: trainingTablePrefix,
    //                                                targetColumn: "target",
    //                                                featureColumns: featureColumns)!
    
    let boostedTreeModel = createBoostedTreeModel(trainingData: trainingTablePrefix,
                                                  validationData: validationTable,
                                                  targetColumn: "target",
                                                  featureColumns: featureColumns)!
    
    //let randomForest = createRandomForestModel(trainingData: trainingTablePrefix,
    //                                           targetColumn: "target",
    //                                           featureColumns: featureColumns)!
    
    //let models: [String : MLRegressable] = ["Linear" : linearModel,
    //                                        "DecisionTree" : decisionTreeModel,
    //                                        "BoostedTree" : boostedTreeModel,
    //                                        "RandomForest" : randomForest]
    
    let models: [String : MLRegressable] = ["BoostedTree" : boostedTreeModel]
    
    // 3. Save model early in case we cut the script short
    for key in models.keys {
        let model = models[key]!
        let outputPath = basePath + "SantanderCreateML/SantanderCreateML/Models/Santander\(key)Regressor.mlmodel"
        print("SAVING MODEL TO \(outputPath)")
        let metadata = MLModelMetadata(author: "Adrian Tineo",
                                       shortDescription: "Predict value for target based on input tabular numeric data",
                                       license: "GPL",
                                       version: "0.1",
                                       additional: nil)
        try model.write(to: URL(fileURLWithPath: outputPath), metadata: metadata)
    }
}


// MARK: test
if evaluate {
    // 1. Import data
    let testSetPath = URL(fileURLWithPath: basePath + "SantanderCreateML/SantanderCreateML/Data/test_pre.csv")
    //let testSetPath = URL(fileURLWithPath: "/Users/adriantineo/dev/ML_data/SantanderCustomerSatisfaction/data/train_head.csv")
    var parsingOptions = MLDataTable.ParsingOptions()
    parsingOptions.delimiter = ","
    parsingOptions.lineTerminator = "\n"
    let testTable = try MLDataTable(contentsOf: testSetPath, options: parsingOptions)
    dump(table: testTable, comment: "TEST TABLE")

    // 2. Load models
    let xgb1 = XgbClassifier1()
    let xgb2 = XgbClassifier2()
    let xgb3 = XgbClassifier3()
    let xgb4 = XgbClassifier4()
    let xgb5 = XgbClassifier5()
    let boostedTree_100_000 = SantanderBoostedTreeRegressor_100_000_it()
    let boostedTree_20_000 = SantanderBoostedTreeRegressor_20_000_it()
    let boostedTree_2_500 = SantanderBoostedTreeRegressor_2_500_it()
    
    // 3. Evaluate models
    print("EVALUATE XGB CLASSIFIERS AVERAGE PREDICTIONS")
    evaluateXgbModels(xgb1: xgb1, xgb2: xgb2, xgb3: xgb3, xgb4:xgb4, xgb5:xgb5, input: testTable)
    
    print("EVALUATE BOOSTED TREE 100_000")
    evaluateBoostedTree_100_000(model: boostedTree_100_000, input: testTable)
    print("EVALUATE BOOSTED TREE 20_000")
    evaluateBoostedTree_20_000(model: boostedTree_20_000, input: testTable)
    print("EVALUATE BOOSTED TREE 2_500")
    evaluateBoostedTree_2_500(model: boostedTree_2_500, input: testTable)
}

let end = CFAbsoluteTimeGetCurrent()
let diff = end - start
print("total script time: \(diff)")
