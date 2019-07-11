//
//  SantanderInferenceTests.swift
//  SantanderInferenceTests
//
//  Created by Adrian Tineo on 24.05.19.
//  Copyright © 2019 adriantineo.com. All rights reserved.
//

import XCTest
import CoreML
@testable import SantanderInference

class SantanderInferenceTests: XCTestCase {
    
    func testBoostedTree_100_000_Performance() {
        // Load data
        let dataURL = Bundle.main.url(forResource: "train", withExtension: "csv")!
        let data = try! String(contentsOf: dataURL)
        let rows = data.split(separator: "\n").map { String($0) }
        let inputs = rows[1...10000].map { makeInputSantanderBoostedTree_100_000(row: $0) }
        
        // Load model
        let boostedTree_100_000 = SantanderBoostedTreeRegressor_100_000_it()
        
        // Make predictions
        let start = CFAbsoluteTimeGetCurrent()
        var numClassZero = 0
        var numClassOne = 0
        var numFailures = 0
        inputs.forEach {input in
            if let prediction = try? boostedTree_100_000.prediction(input: input) {
                if (prediction.target >= 0.5) {
                    numClassOne += 1
                } else {
                    numClassZero += 1
                }
            } else {
                numFailures += 1
            }
        }
        
        let end = CFAbsoluteTimeGetCurrent()
        let diff = end - start
        print("SantanderBoostedTreeRegressor_100_000_it")
        print("Number of predictions for class zero: \(numClassZero)")
        print("Number of predictions for class one: \(numClassOne)")
        print("Number of failed predictions: \(numFailures)")
        print("Average time per prediction is \(diff / Double(inputs.count))")
    }
    
    func testBoostedTree_20_000_Performance(){
        // Load data
        let dataURL = Bundle.main.url(forResource: "train", withExtension: "csv")!
        let data = try! String(contentsOf: dataURL)
        let rows = data.split(separator: "\n").map { String($0) }
        let inputs = rows[1...10000].map { makeInputSantanderBoostedTree_20_000(row: $0) }
        
        // Load model
        let boostedTree_20_000 = SantanderBoostedTreeRegressor_20_000_it()
        
        // Make predictions
        let start = CFAbsoluteTimeGetCurrent()
        var numClassZero = 0
        var numClassOne = 0
        var numFailures = 0
        inputs.forEach {input in
            if let prediction = try? boostedTree_20_000.prediction(input: input) {
                if (prediction.target >= 0.5) {
                    numClassOne += 1
                } else {
                    numClassZero += 1
                }
            } else {
                numFailures += 1
            }
        }
        
        let end = CFAbsoluteTimeGetCurrent()
        let diff = end - start
        print("SantanderBoostedTreeRegressor_20_000_it")
        print("Number of predictions for class zero: \(numClassZero)")
        print("Number of predictions for class one: \(numClassOne)")
        print("Number of failed predictions: \(numFailures)")
        print("Average time per prediction is \(diff / Double(inputs.count))")
    }
    
    func testBoostedTree_2_500_Performance() {
            // Load data
            let dataURL = Bundle.main.url(forResource: "train", withExtension: "csv")!
            let data = try! String(contentsOf: dataURL)
            let rows = data.split(separator: "\n").map { String($0) }
            let inputs = rows[1...10000].map { makeInputSantanderBoostedTree_2_500(row: $0) }
            
            // Load model
            let boostedTree_2_500 = SantanderBoostedTreeRegressor_2_500_it()
            
            // Make predictions
            let start = CFAbsoluteTimeGetCurrent()
            var numClassZero = 0
            var numClassOne = 0
            var numFailures = 0
            inputs.forEach {input in
                if let prediction = try? boostedTree_2_500.prediction(input: input) {
                    if (prediction.target >= 0.5) {
                        numClassOne += 1
                    } else {
                        numClassZero += 1
                    }
                } else {
                    numFailures += 1
                }
            }
            
            let end = CFAbsoluteTimeGetCurrent()
            let diff = end - start
            print("SantanderBoostedTreeRegressor_2_500_it")
            print("Number of predictions for class zero: \(numClassZero)")
            print("Number of predictions for class one: \(numClassOne)")
            print("Number of failed predictions: \(numFailures)")
            print("Average time per prediction is \(diff / Double(inputs.count))")
    }
    
    func testXGBClassifiersPerformance() {
        // Load data
        let dataURL = Bundle.main.url(forResource: "train", withExtension: "csv")!
        let data = try! String(contentsOf: dataURL)
        let rows = data.split(separator: "\n").map { String($0) }
        let inputs1 = rows[1...10000].map { makeInputXgbClassifier1(row: $0) }
        let inputs2 = rows[1...10000].map { makeInputXgbClassifier2(row: $0) }
        let inputs3 = rows[1...10000].map { makeInputXgbClassifier3(row: $0) }
        let inputs4 = rows[1...10000].map { makeInputXgbClassifier4(row: $0) }
        let inputs5 = rows[1...10000].map { makeInputXgbClassifier5(row: $0) }
        
        // Load models
        let xgb1 = XgbClassifier1()
        let xgb2 = XgbClassifier2()
        let xgb3 = XgbClassifier3()
        let xgb4 = XgbClassifier4()
        let xgb5 = XgbClassifier5()
        
        // Make predictions
        let start = CFAbsoluteTimeGetCurrent()
        var numClassZero = 0
        var numClassOne = 0
        var numFailures = 0
        for idx in 0..<inputs1.count {
            do {
                let input1 = inputs1[idx]
                let predictionXgb1 = (try xgb1.prediction(input: input1)).target
                let input2 = inputs2[idx]
                let predictionXgb2 = (try xgb2.prediction(input: input2)).target
                let input3 = inputs3[idx]
                let predictionXgb3 = (try xgb3.prediction(input: input3)).target
                let input4 = inputs4[idx]
                let predictionXgb4 = (try xgb4.prediction(input: input4)).target
                let input5 = inputs5[idx]
                let predictionXgb5 = (try xgb5.prediction(input: input5)).target
                
                let sum = predictionXgb1 + predictionXgb2 + predictionXgb3 + predictionXgb4 + predictionXgb5
                
                if sum > 2 {
                    numClassOne += 1
                } else {
                    numClassZero += 1
                }
            } catch {
                numFailures += 1
            }
        }
        
        let end = CFAbsoluteTimeGetCurrent()
        let diff = end - start
        print("XGBClassifiers")
        print("Number of predictions for class zero: \(numClassZero)")
        print("Number of predictions for class one: \(numClassOne)")
        print("Number of failed predictions: \(numFailures)")
        print("Average time per prediction is \(diff / Double(inputs1.count))")
    }

}
