#include <stdio.h> 

int calculatesum(int a, int b){
  return a + b;
}

int main() {
  bool thing = true;

  for (int i=0; i<4; i++){
    if (thing) {
      printf("lets fucking goooo %d \n", calculatesum(4, i)); 
    }
  }
  return 0;
}
