package nlp_test

import (
	"fmt"
	"testing"

	"github.com/Shixzie/nlp"
)

func TestP(t *testing.T) {
	type Song struct {
		Name   string
		Artist string
		Likes  int
	}

	songSamples := []string{
		"play {Name} by {Artist}",
		"play {Name} from {Artist}",
		"play {Name}",
		"from {Artist} play {Name}",
	}

	type Order struct {
		Product  string
		Quantity int
	}

	orderSamples := []string{
		"dame {Quantity}, {Product}",
		"ordena {Quantity}, {Product}",
		"compra un {Product}",
		"compra {Quantity}, {Product}",
		"compra {Quantity} de {Product}",
	}

	nl := nlp.New()

	err := nl.RegisterModel(Order{}, orderSamples)
	if err != nil {
		panic(err)
	}
	err = nl.RegisterModel(Song{}, songSamples)
	if err != nil {
		panic(err)
	}

	err = nl.Learn() // you must call Learn after all models are registered and before calling P
	if err != nil {
		panic(err)
	}
	expr := fmt.Sprintf("COMPRA 250 de cajas vacías")
	o := nl.P(expr) // after learning you can call P the times you want
	if order, ok := o.(Order); ok {
		fmt.Println("Success")
		fmt.Printf("%#v\n", order)
	} else {
		fmt.Println("Failed")
		fmt.Printf("%#v\n", o)
	}

	// Prints
	//
	// Success
	// nlp_test.Order{Product: "King", Quantity: 25}
}
