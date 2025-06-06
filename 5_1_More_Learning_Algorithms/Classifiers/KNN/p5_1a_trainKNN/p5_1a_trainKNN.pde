//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************

import Weka4P.*;
Weka4P wp;

void setup() {
  size(500, 500);             //set a canvas
  frameRate(60);
  wp = new Weka4P(this);
  wp.loadTrainARFF("mouseTrain.arff"); //load a ARFF dataset
  wp.trainKNN(1);             //train a SV classifier with K = 1
  wp.setModelDrawing(2);         //set the model visualization (for 2D features) with unit = 2
  wp.evaluateTrainSet(5, false, true);  //5-fold cross validation
  wp.saveModel("KNN.model"); //save the model
}

void draw() {
  wp.drawModel(0, 0); //draw the model visualization (for 2D features)
  wp.drawDataPoints(wp.train); //draw the datapoints
  float[] X = {mouseX, mouseY}; 
  String Y = wp.getPrediction(X);
  wp.drawPrediction(X, Y); //draw the prediction
}
