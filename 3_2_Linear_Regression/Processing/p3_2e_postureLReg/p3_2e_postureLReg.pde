//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************
import Weka4P.*;
Weka4P wp;

import processing.serial.*;
Serial port; 

int sensorNum = 3;
int[] rawData = new int[sensorNum];
boolean dataUpdated = false;

void setup() {
  size(500, 500);             //set a canvas
  frameRate(60);
  wp = new Weka4P(this);
  
  //Initialize the serial port
  for (int i = 0; i < Serial.list().length; i++) println("[", i, "]:", Serial.list()[i]);
  String portName = Serial.list()[Serial.list().length-1];//MAC: check the printed list
  //String portName = Serial.list()[9];//WINDOWS: check the printed list
  port = new Serial(this, portName, 115200);
  port.bufferUntil('\n'); // arduino ends each data packet with a carriage return 
  port.clear();           // flush the Serial buffer

  wp.loadTrainARFF("accData.arff"); //load a ARFF dataset
  wp.trainLinearRegression();               //train a regressor
  wp.saveModel("postureLReg.model"); //save the model
  
  background(52);
}

void draw() {
  if (dataUpdated) {
    background(52);
    fill(255);
    float[] X = {rawData[0], rawData[1], rawData[2]}; 
    double Y = wp.getPredictionIndex(X);
    textSize(32);
    textAlign(CENTER,CENTER);
    String text = "Prediction: "+nf((float)Y,0,2)+
                  "\n X="+rawData[0]+
                  "\n Y="+rawData[1]+
                  "\n Z="+rawData[2];
    text(text, width/2, height/2);
    if(Y>0.5){
      port.write('a');
    }else{
      port.write('b');
    }
    dataUpdated = false;
  }
}

void serialEvent(Serial port) {   
  String inData = port.readStringUntil('\n');  // read the serial string until seeing a carriage return
  if (!dataUpdated) 
  {
    if (inData.charAt(0) == 'A') {
      rawData[0] = int(trim(inData.substring(1)));
    }
    if (inData.charAt(0) == 'B') {
      rawData[1] = int(trim(inData.substring(1)));
    }
    if (inData.charAt(0) == 'C') {
      rawData[2] = int(trim(inData.substring(1)));
      dataUpdated = true;
    }
  }
  return;
}
