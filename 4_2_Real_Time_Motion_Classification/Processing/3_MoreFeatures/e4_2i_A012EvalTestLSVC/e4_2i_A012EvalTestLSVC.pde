//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************

import Weka4P.*;
Weka4P wp;

void setup() {
  size(500, 500, P2D);
  frameRate(60);
  wp = new Weka4P(this);
  wp.loadTrainARFF("A012GestTrain.arff");//load a ARFF dataset
  wp.loadTestARFF("A012GestTest.arff");//load a ARFF dataset
  wp.loadModel("LinearSVC.model"); //load a pretrained model.
  wp.evaluateTestSet(false, true);  //5-fold cross validation (isRegression = false, showEvalDetails=true)
}

void draw() {
  background(255);
}
