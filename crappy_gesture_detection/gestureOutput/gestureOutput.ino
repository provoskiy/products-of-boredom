#include <Wire.h>
#include <LiquidCrystal.h>

LiquidCrystal lcd(7, 8, 9, 10, 11, 12);

int data;

String incoming = "";

void setup() {
  Serial.begin(9600);
  lcd.begin(16, 2);
}

void loop() {
  while(Serial.available()){
    char data = Serial.read();
    
    if (data == '\n'){
      lcd.clear();
      lcd.setCursor(0,0);
      lcd.print(incoming);
      incoming = "";
    }
    else{
      incoming += data;
    }
    delay(100);
  }
}

