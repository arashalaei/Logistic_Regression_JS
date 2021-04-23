/* jslint esversion : 9*/
/**
 * @author Arash Alaei <arashalaei22@gmail.com>
 * @since 02/00/2021
 */

// Importing libraries
const dfd = require('danfojs-node');
const tf = require('@tensorflow/tfjs');
const LogisticRegression = require('ml-logistic-regression');
const path = require("path");
const {Matrix} = require('ml-matrix');
const ConfusionMatrix = require('ml-confusion-matrix');

(async () =>{
    // Importing dataset
    const dataset = await dfd.read_csv(`file://${path.join(__dirname, './Social_Network_Ads.csv')}`);
    const x = dataset.iloc({rows:[':'], columns:[':2']}).values;
    const y = tf.util.flatten(dataset.iloc({rows:[':'], columns:['2']}).values);
    // Spliting dataset into train & test set
    let x_train = [], x_test = [], y_train = [], y_test = [];
    let x_shuff = [...x];
    let y_shuff = [...y];
    shuffleCombo(x_shuff, y_shuff);

    x_train = [...x_shuff.slice(0, Math.floor(0.8 * x.length))];
    x_test  = [...x_shuff.slice(Math.floor(0.8 * x.length))];

    y_train = [...y_shuff.slice(0, Math.floor(0.8 * y.length))];
    y_test  = [...y_shuff.slice(Math.floor(0.8 * y.length))];
    // Feature scaling
    let df = new dfd.DataFrame(x_train);
    let scaler = new dfd.StandardScaler();
    scaler.fit(df.iloc({rows:[':'],columns:[":"]}));
    let df_enc = scaler.transform(df.iloc({rows:[':'],columns:[":"]}));
    x_train = df_enc.values;

    df = new dfd.DataFrame(x_test);
    scaler = new dfd.StandardScaler();
    scaler.fit(df.iloc({rows:[':'],columns:[":"]}));
    df_enc = scaler.transform(df.iloc({rows:[':'],columns:[":"]}));
    x_test = df_enc.values;
    // Training the logistic regression model on the training set
    const logreg = new LogisticRegression({ numSteps: 1000, learningRate: 5e-3 });
    logreg.train(new Matrix(x_train), Matrix.columnVector(y_train));
    // predicting a new result
    df = new dfd.DataFrame([[30, 87000]]);
    console.log(logreg.predict(new Matrix(scaler.transform(df).values)));
    // Predicting the test set result 
    let y_pred = logreg.predict(new Matrix(x_test));

    const CM = ConfusionMatrix.fromLabels(y_test, y_pred);
    console.log(CM.getAccuracy());
})();


function shuffleCombo(array, array2) {
    
    if (array.length !== array2.length) {
      throw new Error(
        `Array sizes must match to be shuffled together ` +
        `First array length was ${array.length}` +
        `Second array length was ${array2.length}`);
    }
    let counter = array.length;
    let temp, temp2;
    let index = 0;
    // While there are elements in the array
    while (counter > 0) {
      // Pick a random index
      index = (Math.random() * counter) | 0;
      // Decrease counter by 1
      counter--;
      // And swap the last element of each array with it
      temp = array[counter];
      temp2 = array2[counter];
      array[counter] = array[index];
      array2[counter] = array2[index];
      array[index] = temp;
      array2[index] = temp2;
    }
  }
  