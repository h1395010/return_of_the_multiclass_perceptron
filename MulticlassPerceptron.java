
import java.text.DecimalFormat;
import java.util.Set;

import com.google.common.collect.Table;
import com.google.common.collect.Table.Cell;

public class MulticlassPerceptron 
{
	  static int MAX_ITER = 100;
	  static double LEARNING_RATE = 0.1;           
	  static int theta = 0; 
	  
      static final String LABEL_atheism = "atheism";
      static final String LABEL_sports = "sports";
      static final String LABEL_science = "science";
      static final String LABEL_politics = "politics";
	  
	  public static void perceptron( Table< int[] , String , Integer > train_freq_count_against_globo_dict,
			  					     Table< int[] , String , Integer > test_freq_count_against_globo_dict,
			  						 Set<String> GLOBO_DICT )
	  {
		  int globo_dict_size = GLOBO_DICT.size();
		  int number_of_files__train = train_freq_count_against_globo_dict.size();
		  
		  double[] weights__science = new double[ globo_dict_size + 1 ];//one for bias
		  double[] weights__sports = new double[ globo_dict_size + 1 ];//one for bias
		  double[] weights__politics = new double[ globo_dict_size + 1 ];//one for bias
		  double[] weights__atheism = new double[ globo_dict_size + 1 ];//one for bias
		  
		  for (int i = 0; i < ( globo_dict_size + 1 ); i++) 
		  {
            weights__science[i] = randomNumber(0,1);
            weights__sports[i] = randomNumber(0,1);
            weights__politics[i] = randomNumber(0,1);
            weights__atheism[i] = randomNumber(0,1);
		  }	    

		   double[][] feature_matrix__train = new double[ number_of_files__train ][ globo_dict_size ];
		   int[] outputs__science = new int [ number_of_files__train ];
		   int[] outputs__sports = new int [ number_of_files__train ];
		   int[] outputs__politics = new int [ number_of_files__train ];
		   int[] outputs__atheism = new int [ number_of_files__train ];
		    
		   int z = 0;
		   for ( Cell< int[] , String , Integer > cell: train_freq_count_against_globo_dict.cellSet() )
		   {			   
			   int[] container_of_feature_vector = cell.getRowKey();
			   
			   for (int q = 0; q < globo_dict_size; q++) 
	           {
				   feature_matrix__train[z][q] = container_of_feature_vector[q];
	           }
			   outputs__science[z] =  String.valueOf( cell.getColumnKey() ).equals(LABEL_atheism)  ||
					                  String.valueOf( cell.getColumnKey() ).equals(LABEL_sports)   ||
					                  String.valueOf( cell.getColumnKey() ).equals(LABEL_politics) ? 0 : 1;
			   
			   outputs__sports[z] =   String.valueOf( cell.getColumnKey() ).equals(LABEL_atheism)  ||
		                 			  String.valueOf( cell.getColumnKey() ).equals(LABEL_science)  ||
		                 			  String.valueOf( cell.getColumnKey() ).equals(LABEL_politics) ? 0 : 1;
			   
			   outputs__politics[z] = String.valueOf( cell.getColumnKey() ).equals(LABEL_atheism)  ||
		                 			  String.valueOf( cell.getColumnKey() ).equals(LABEL_sports)   ||
		                 			  String.valueOf( cell.getColumnKey() ).equals(LABEL_science)  ? 0 : 1;
			   
			   outputs__atheism[z] =  String.valueOf( cell.getColumnKey() ).equals(LABEL_science)  ||
		                 			  String.valueOf( cell.getColumnKey() ).equals(LABEL_sports)   ||
		                 			  String.valueOf( cell.getColumnKey() ).equals(LABEL_politics) ? 0 : 1;
	           z++;
		   }
		    
		  //LEARNING WEIGHTS
		  double localError, globalError;
		  int p, iteration, output;
		
		  //SCIENCE
		  System.out.println("SCIENCE vs. ALL:");
		  iteration = 0;
		  do 
		  {
			  iteration++;
			  globalError = 0;
			  //loop through all instances (complete one epoch)
			  for (p = 0; p < number_of_files__train; p++) 
			  {
				  // calculate predicted class
				  output = calculateOutput( theta, weights__science, feature_matrix__train, p, globo_dict_size );
				  // difference between predicted and actual class values
				  localError = outputs__science[p] - output;
				  //update weights and bias
				  for (int i = 0; i < globo_dict_size; i++) 
				  {
					  weights__science[i] += ( LEARNING_RATE * localError * feature_matrix__train[p][i] );
				  }
				  weights__science[ globo_dict_size ] += ( LEARNING_RATE * localError );
				  
				  //summation of squared error (error value for all instances)
				  globalError += (localError*localError);
			  }

			  /* Root Mean Squared Error */
			  if (iteration < 10) 
				  System.out.println("Iteration 0" + iteration + " : RMSE = " + Math.sqrt( globalError/number_of_files__train ) );
			  else
				  System.out.println("Iteration " + iteration + " : RMSE = " + Math.sqrt( globalError/number_of_files__train ) );
			  //System.out.println( Arrays.toString( weights ) );
		  } 
		  while(globalError != 0 && iteration<=MAX_ITER);
		  
		  



		  //SPORTS
		  System.out.println("SPORTS vs. ALL");
		  iteration = 0;
		  do 
		  {
			  iteration++;
			  globalError = 0;
			  //loop through all instances (complete one epoch)
			  for (p = 0; p < number_of_files__train; p++) 
			  {
				  // calculate predicted class
				  output = calculateOutput( theta, weights__sports, feature_matrix__train, p, globo_dict_size );
				  // difference between predicted and actual class values
				  localError = outputs__sports[p] - output;
				  //update weights and bias
				  for (int i = 0; i < globo_dict_size; i++) 
				  {
					  weights__sports[i] += ( LEARNING_RATE * localError * feature_matrix__train[p][i] );
				  }
				  weights__sports[ globo_dict_size ] += ( LEARNING_RATE * localError );
				  
				  //summation of squared error (error value for all instances)
				  globalError += (localError*localError);
			  }

			  /* Root Mean Squared Error */
			  if (iteration < 10) 
				  System.out.println("Iteration 0" + iteration + " : RMSE = " + Math.sqrt( globalError/number_of_files__train ) );
			  else
				  System.out.println("Iteration " + iteration + " : RMSE = " + Math.sqrt( globalError/number_of_files__train ) );
			  //System.out.println( Arrays.toString( weights ) );
		  } 
		  while(globalError != 0 && iteration<=MAX_ITER);
		  



		  
		  //POLITICS
		  System.out.println("POLITICS vs. ALL");
		  iteration = 0;
		  do 
		  {
			  iteration++;
			  globalError = 0;
			  //loop through all instances (complete one epoch)
			  for (p = 0; p < number_of_files__train; p++) 
			  {
				  // calculate predicted class
				  output = calculateOutput( theta, weights__politics, feature_matrix__train, p, globo_dict_size );
				  // difference between predicted and actual class values
				  localError = outputs__politics[p] - output;
				  //update weights and bias
				  for (int i = 0; i < globo_dict_size; i++) 
				  {
					  weights__politics[i] += ( LEARNING_RATE * localError * feature_matrix__train[p][i] );
				  }
				  weights__politics[ globo_dict_size ] += ( LEARNING_RATE * localError );
				  
				  //summation of squared error (error value for all instances)
				  globalError += (localError*localError);
			  }

			  /* Root Mean Squared Error */
			  if (iteration < 10) 
				  System.out.println("Iteration 0" + iteration + " : RMSE = " + Math.sqrt( globalError/number_of_files__train ) );
			  else
				  System.out.println("Iteration " + iteration + " : RMSE = " + Math.sqrt( globalError/number_of_files__train ) );
			  //System.out.println( Arrays.toString( weights ) );
		  } 
		  while(globalError != 0 && iteration<=MAX_ITER);
		  
		  


		  //ATHEISM
		  System.out.println("ATHEISM vs. ALL");
		  iteration = 0;
		  do 
		  {
			  iteration++;
			  globalError = 0;
			  //loop through all instances (complete one epoch)
			  for (p = 0; p < number_of_files__train; p++) 
			  {
				  // calculate predicted class
				  output = calculateOutput( theta, weights__atheism, feature_matrix__train, p, globo_dict_size );
				  // difference between predicted and actual class values
				  localError = outputs__atheism[p] - output;
				  //update weights and bias
				  for (int i = 0; i < globo_dict_size; i++) 
				  {
					  weights__atheism[i] += ( LEARNING_RATE * localError * feature_matrix__train[p][i] );
				  }
				  weights__atheism[ globo_dict_size ] += ( LEARNING_RATE * localError );
				  
				  //summation of squared error (error value for all instances)
				  globalError += (localError*localError);
			  }

			  /* Root Mean Squared Error */
			  if (iteration < 10) 
				  System.out.println("Iteration 0" + iteration + " : RMSE = " + Math.sqrt( globalError/number_of_files__train ) );
			  else
				  System.out.println("Iteration " + iteration + " : RMSE = " + Math.sqrt( globalError/number_of_files__train ) );
			  //System.out.println( Arrays.toString( weights ) );
		  } 
		  while(globalError != 0 && iteration<=MAX_ITER);
		  
		  


		  	  
		  
		  //TEST   
	       int number_of_files__test = test_freq_count_against_globo_dict.size();
	       double[][] feature_matrix__test = new double[ number_of_files__test ][ globo_dict_size ];
	        
		   String[] test_file_true_label = new String [ number_of_files__test ];
		    
		   int x = 0;
		   for ( Cell< int[] , String , Integer > cell: test_freq_count_against_globo_dict.cellSet() )
		   {			   
			   int[] container_of_feature_vector__test = cell.getRowKey();
			   
			   for (int q = 0; q < globo_dict_size; q++) 
	           {
				   feature_matrix__test[x][q] = container_of_feature_vector__test[q];
	           }
			   test_file_true_label[x] = (String)( cell.getColumnKey() );
	           
	           x++;
		   }
		   System.out.println();
		   
		   
		   double tp_science = 0.0;
		   double fp_science = 0.0; 
		   double tn_science = 0.0;
		   double fn_science = 0.0;
		   
		   double precision_science = 0.0;
		   double recall_science = 0.0;
		   double f_measure_science = 0.0;
		   
		   double tp_sports = 0.0;
		   double fp_sports = 0.0; 
		   double tn_sports = 0.0;
		   double fn_sports = 0.0;
		   
		   double precision_sports = 0.0;
		   double recall_sports = 0.0;
		   double f_measure_sports = 0.0;
		   
		   double tp_atheism = 0.0;
		   double fp_atheism = 0.0; 
		   double tn_atheism = 0.0;
		   double fn_atheism = 0.0;
		   
		   double precision_atheism = 0.0;
		   double recall_atheism = 0.0;
		   double f_measure_atheism = 0.0;
		   
		   double tp_politics = 0.0;
		   double fp_politics = 0.0; 
		   double tn_politics = 0.0;
		   double fn_politics = 0.0;
		   
		   double precision_politics = 0.0;
		   double recall_politics = 0.0;
		   double f_measure_politics = 0.0;
		   
		   
		   int actual_class = 0;
		   
		  for (p = 0; p < number_of_files__test; p++) 
		  {
			  
			  int predict_science = calculateOutput( theta, weights__science, feature_matrix__test, p, globo_dict_size );
			  
		      actual_class = ( test_file_true_label[p] ).equals( LABEL_science ) ? 1 : 0;
			  
		      if( actual_class == 1 && predict_science == 1 )
		    	  tp_science++;
		      if( actual_class == 1 && predict_science == 0 )
		    	  fn_science++;
		      if( actual_class == 0 && predict_science == 1 )
		    	  fp_science++;
		      if( actual_class == 0 && predict_science == 0 )
		    	  tn_science++;  

			  int predict_sports = calculateOutput( theta, weights__sports, feature_matrix__test, p, globo_dict_size );
			  
			  actual_class = ( test_file_true_label[p] ).equals( LABEL_sports ) ? 1 : 0;
			  
		      if( actual_class == 1 && predict_sports == 1 )
		    	  tp_sports++;
		      if( actual_class == 1 && predict_sports == 0 )
		    	  fn_sports++;
		      if( actual_class == 0 && predict_sports == 1 )
		    	  fp_sports++;
		      if( actual_class == 0 && predict_sports == 0 )
		    	  tn_sports++;

			  int predict_politics = calculateOutput( theta, weights__politics, feature_matrix__test, p, globo_dict_size );
			  
			  actual_class = ( test_file_true_label[p] ).equals( LABEL_politics ) ? 1 : 0;
			  
		      if( actual_class == 1 && predict_politics == 1 )
		    	  tp_politics++;
		      if( actual_class == 1 && predict_politics == 0 )
		    	  fn_politics++;
		      if( actual_class == 0 && predict_politics == 1 )
		    	  fp_politics++;
		      if( actual_class == 0 && predict_politics == 0 )
		    	  tn_politics++;

			  int predict_atheism = calculateOutput( theta, weights__atheism, feature_matrix__test, p, globo_dict_size );
			  
			  actual_class = ( test_file_true_label[p] ).equals( LABEL_atheism ) ? 1 : 0;
			  
		      if( actual_class == 1 && predict_atheism == 1 )
		    	  tp_atheism++;
		      if( actual_class == 1 && predict_atheism == 0 )
		    	  fn_atheism++;
		      if( actual_class == 0 && predict_atheism == 1 )
		    	  fp_atheism++;
		      if( actual_class == 0 && predict_atheism == 0 )
		    	  tn_atheism++; 
  
		  }
		  
		  System.out.println( "SCIENCE");
		  System.out.println( "tp_science: " + tp_science );
		  System.out.println( "fp_science: " + fp_science );
		  System.out.println( "tn_science: " + tn_science );
		  System.out.println( "fn_science: " + fn_science );
		  System.out.println();
		  
		  precision_science = tp_science / (tp_science + fp_science);
		  System.out.println( "precision_science = " + precision_science );
		  
		  recall_science = tp_science / (tp_science + fn_science);
		  System.out.println( "recall_science = " + recall_science );
		  
		  f_measure_science = ( 2 * ( precision_science * recall_science ) ) / ( precision_science + recall_science );
		  System.out.println( "f_measure_science = " + f_measure_science );
		  System.out.println();
		  System.out.println();
		  System.out.println();
		  
		  
		  
		  
		  System.out.println( "SPORTS");
		  System.out.println( "tp_sports: " + tp_sports );
		  System.out.println( "fp_sports: " + fp_sports );
		  System.out.println( "tn_sports: " + tn_sports );
		  System.out.println( "fn_sports: " + fn_sports );
		  System.out.println();
		  
		  precision_sports = tp_sports / (tp_sports + fp_sports);
		  System.out.println( "precision_sports = " + precision_sports );
		  
		  recall_sports = tp_sports / (tp_sports + fn_sports);
		  System.out.println( "recall_sports = " + recall_sports );
		  
		  f_measure_sports = ( 2 * ( precision_sports * recall_sports ) ) / ( precision_sports + recall_sports );
		  System.out.println( "f_measure_sports = " + f_measure_sports );
		  System.out.println();
		  System.out.println();
		  System.out.println();
		  
		  
		  
		  System.out.println( "POLITICS");
		  System.out.println( "tp_politics: " + tp_politics );
		  System.out.println( "fp_politics: " + fp_politics );
		  System.out.println( "tn_politics: " + tn_politics );
		  System.out.println( "fn_politics: " + fn_politics );
		  System.out.println();
		  
		  precision_politics = tp_politics / (tp_politics + fp_politics);
		  System.out.println( "precision_politics = " + precision_politics );
		  
		  recall_politics = tp_politics / (tp_politics + fn_politics);
		  System.out.println( "recall_politics = " + recall_politics );
		  
		  f_measure_politics = ( 2 * ( precision_politics * recall_politics ) ) / ( precision_politics + recall_politics );
		  System.out.println( "f_measure_politics = " + f_measure_politics );
		  System.out.println();
		  System.out.println();
		  System.out.println();
		  
		  
		  
		  System.out.println( "ATHEISM");
		  System.out.println( "tp_atheism: " + tp_atheism );
		  System.out.println( "fp_atheism: " + fp_atheism );
		  System.out.println( "tn_atheism" + tn_atheism );
		  System.out.println( "fn_atheism: " + fn_atheism );
		  System.out.println();
		  
		  precision_atheism = tp_atheism / (tp_atheism + fp_atheism);
		  System.out.println( "precision_atheism = " + precision_atheism );
		  
		  recall_atheism = tp_atheism / (tp_atheism + fn_atheism);
		  System.out.println( "recall_atheism = " + recall_atheism );
		  
		  f_measure_atheism = ( 2 * ( precision_atheism * recall_atheism ) ) / ( precision_atheism + recall_atheism );
		  System.out.println( "f_measure_atheism = " + f_measure_atheism );
		  System.out.println();
		  System.out.println();
		  System.out.println();
		  
		  
		
		  System.out.println("------------------------------------------------------------------------");
		  System.out.println();
		  
		  
		  //average them!
		  System.out.println( "AGREGATE");
		  double precision = ( ( precision_atheism + precision_sports + precision_science + precision_politics ) / 4 );
		  System.out.println( "precision = " + precision );
		  
		  double recall =  ( ( recall_atheism + recall_sports + recall_science + recall_politics ) / 4 );
		  System.out.println( "recall = " + recall );
		  
		  double f_measure =  ( ( f_measure_atheism + f_measure_sports + f_measure_science + f_measure_politics ) / 4 );
		  System.out.println( "f_measure = " + f_measure );
		  
				  

	  }
	  
	  static int calculateOutput( int theta, double weights[], double[][] feature_matrix, int file_index, int globo_dict_size )
	  {
	     //double sum = x * weights[0] + y * weights[1] + z * weights[2] + weights[3];
		 double sum = 0;
		 
		 for (int i = 0; i < globo_dict_size; i++) 
		 {
			 sum += ( weights[i] * feature_matrix[file_index][i] );
		 }
		 //bias
		 sum += weights[ globo_dict_size ];
		 
	     return (sum >= theta) ? 1 : 0;
	  }
	  
	  public static double randomNumber(int min , int max) 
	  {
		  DecimalFormat df = new DecimalFormat("#.####");
		  double d = min + Math.random() * (max - min);
		  String s = df.format(d);
		  double x = Double.parseDouble(s);
		  return x;
	  }
}
