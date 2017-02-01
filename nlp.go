// Package nlp provides general purpose Natural Language Processing in any-lang.
package nlp

import (
	"bytes"
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/text"
)

// allowed punctuation symbols that can be by the right side of a keyword. eg:
// {Keyword},
var punctuation = []string{",", ".", "!", "¡", "-", "_", ";", ":", "]", "[", "'", "?", "=", "^", "|", `"`, "`"}

// GUIDE FOR CONTRIBUTORS
//
// Simple resume of how it internally works:
// New()
// - Simply creates a *NL with the output beign an empty buffer
//
// RegisterModel()
// - Checks that i is a struct
// - The model name is the same as the i type
// - Scan all fields:
//   - It'll add a field to model.fields only if the field type is either int or string
//     and if the field is exported
// - Adds the model to the slice of model inside the NL type
// (more info about model type check type definition)
//
// Learn()
// - Calls the learn() func for every model inside the NL.models slice
// - Trains the NaiveBayes algorithm that will be used to choose which model to use
//   when calling NL.P() (the training is based on all model samples)
//
// model.learn()
// - iterates through all samples searching for keywords
// - keywords must be inside {} eg. {Keyword}
//   (a keyword must have the same name as a struct field for that specific model)
// - a key type is created and added to the model.keys slice
// (more info about key type check type definition)
//
// Things to do:
// Tests:
// - unit tests
//
// Limits:
// - model.learn() should just search for limits
//
// Feedback:
// - implement a NL.Feedback() func
//   - this would take some chan of nlp.Feed or something
//   - a Feed type would have to be created
//     - would contain the model name, an expr and the name of the struct field that expr belongs to
//
// Keywords recognition:
// - implement some kind of stemmer for both model.learn() and model.fit()
// - the model type should have it's own *text.NaiveBayes
//   - this field should be called naive
//   - it would be trained with the stemmed words
//   - it would be used to recognize wheter a word belongs to a specific of keyword or not

// NL is a Natural Language Processor
type NL struct {
	models []*model
	naive  *text.NaiveBayes
	// Output contains the training output for the
	// NaiveBayes algorithm
	Output *bytes.Buffer
}

// New returns a *NL
func New() *NL { return &NL{Output: bytes.NewBufferString("")} }

// P proccesses the expr and returns one of
// the types passed as the i parameter to the RegisterModel
// func filled with the data inside expr
func (nl *NL) P(expr string) interface{} { return nl.models[nl.naive.Predict(expr)].fit(expr) }

// Learn maps the models samples to the models themselves and
// returns an error if something occurred while learning
func (nl *NL) Learn() error {
	if len(nl.models) > 0 {
		// Create a new NaiveBayes algorithm
		stream := make(chan base.TextDatapoint)
		errors := make(chan error)
		nl.naive = text.NewNaiveBayes(stream, uint8(len(nl.models)), base.OnlyWordsAndNumbers)
		nl.naive.Output = nl.Output
		go nl.naive.OnlineLearn(errors)
		for modelIndice := range nl.models {
			// make a model learn
			err := nl.models[modelIndice].learn()
			if err != nil {
				return err
			}
			for _, s := range nl.models[modelIndice].samples {
				// the class value (Y) is the indice of the model inside the slice
				stream <- base.TextDatapoint{
					X: s,
					Y: uint8(modelIndice),
				}
			}
		}
		close(stream)
		for {
			err := <-errors
			if err != nil {
				return fmt.Errorf("error occurred while learning: %s", err)
			}
			// training is done!
			break
		}
		return nil
	}
	return fmt.Errorf("register at least one model before learning")
}

type model struct {
	name    string
	tpy     reflect.Type
	fields  []field
	keys    [][]key
	samples []string
}

// field contains info about a struct field
// such as it's index in the struct type,
// the name of the field and the kind (int, string)
type field struct {
	index int
	name  string
	kind  reflect.Kind
}

// key represents a keyword inside a sample,
// it contains the side words of the keyword (limits)
// (left, right) and the word itself (name of the struct field)
// plus the ID (indice in the slice) (model.samples, model.fields)
// of the sample and field
type key struct {
	left, word, right string
	sampleID, fieldID int
	prob              float32
}

// RegisterModel registers a model i and creates possible patterns
// from samples.
//
// NOTE: samples must have a special formatting, see example below.
//
func (nl *NL) RegisterModel(i interface{}, samples []string) error {
	if i == nil {
		return fmt.Errorf("can't create model from nil value")
	}
	if len(samples) == 0 {
		return fmt.Errorf("samples can't be nil or empty")
	}
	tpy, val := reflect.TypeOf(i), reflect.ValueOf(i)
	if tpy.Kind() == reflect.Struct {
		mod := &model{
			name:    tpy.Name(),
			tpy:     tpy,
			samples: samples,
			keys:    make([][]key, len(samples)),
		}
	NextField:
		for i := 0; i < tpy.NumField(); i++ {
			if tpy.Field(i).Anonymous || tpy.Field(i).PkgPath != "" {
				continue NextField
			}
			switch val.Field(i).Kind() {
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.String:
				mod.fields = append(mod.fields, field{i, tpy.Field(i).Name, val.Field(i).Kind()})
			}
		}
		nl.models = append(nl.models, mod)
		return nil
	}
	return fmt.Errorf("can't create model from non-struct type")
}

func (m *model) learn() error {
	isKeyword := func(word string) bool {
		if len(word) == 1 || len(word) == 0 {
			return false
		}
		if string(word[0]) == "{" && string(word[len(word)-1]) == "}" {
			return true
		}
		if string(word[0]) == "{" && string(word[len(word)-2]) == "}" {
			return true
		}
		return false
	}
	for sid, s := range m.samples {
		badWords := strings.Split(s, " ")
		words := []string{}
		for _, bw := range badWords {
			added := false
			for _, p := range punctuation {
				if strings.HasSuffix(bw, p) {
					words = append(words, string(bw[:len(bw)-1]))
					words = append(words, string(bw[len(bw)-1]))
					added = true
				}
			}
			if !added {
				words = append(words, bw)
			}
		}
		wsLen := len(words)
		for wordID, word := range words {
			if isKeyword(word) {
				var left, kword, right, replace string
				var fieldID int
				replace = "-"
				keyword := word[1 : len(word)-1]
				kl := len(m.keys[sid])
				for fid, f := range m.fields {
					if f.name == keyword {
						fieldID = fid
						if wordID == 0 { // {} <- first (the keyword being the first word in the sample)
							if wordID == wsLen-1 { // {} (and being the only word in the sample)
								kword = keyword
							} else { // {} ...
								if isKeyword(words[wordID+1]) { // {X} {Y} <- X (and having a keyword to the right)
									kword = keyword
									right = replace
								} else { // {} ... (and having a limit to the right)
									kword = keyword
									right = words[wordID+1]
								}
							}
						} else {
							if wordID == wsLen-1 { // ... {} <- last (the keyword being the last word in the sample)
								if isKeyword(words[wordID-1]) { // {X} {Y} <- Y (and having a keyword to the left)
									left = replace
									kword = keyword
								} else { // ... {} (and having a limit to the left)
									left = words[wordID-1]
									kword = keyword
								}
							} else { // ... {} ... (the keyword being somewhere in the sample)
								if isKeyword(words[wordID-1]) { // ... {X} {Y} ... <- Y (and having a keyword to the left)
									if isKeyword(words[wordID+1]) { // ... {X} {Y} {Z} ... <- Y (and having keywords on both sides)
										left = replace
										kword = keyword
										right = replace
									} else { // ... {X} {Y} ... <- Y (and having a keyword on the left)
										left = replace
										kword = keyword
										right = words[wordID+1]
									}
								} else if isKeyword(words[wordID+1]) { // ... {X} {Y} ... <- X (and having a keyword to the right)
									left = words[wordID-1]
									kword = keyword
									right = replace
								} else { // ... {} ... (and having limits on both sides)
									left = words[wordID-1]
									kword = keyword
									right = words[wordID+1]
								}
							}
						}
					}
				}
				k := key{
					left:     left,
					right:    right,
					word:     kword,
					sampleID: sid,
					fieldID:  fieldID,
				}
				if k.word != "" {
					m.keys[sid] = append(m.keys[sid], k)
				}
				if len(m.keys[sid]) == kl {
					return fmt.Errorf("error while processing model samples, miss-spelled '%s'", keyword)
				}
			}
		}
	}
	return nil
}

func (m *model) selectBestSample(expr string) (int, map[string][]int) {
	// map[sample_id]score
	scores := make(map[int]int)
	// map[sample_id]map[keyword]indices
	// indices is a []int of len = 2 that contains
	// the indices where a keyword starts and ends on the expr
	wordsMap := make(map[int]map[string][]int)
	// expr splitted by Space
	badWords := strings.Split(expr, " ")
	words := []string{}
	for _, bw := range badWords {
		added := false
		for _, p := range punctuation {
			if strings.HasSuffix(bw, p) {
				words = append(words, string(bw[:len(bw)-1]))
				words = append(words, string(bw[len(bw)-1]))
				added = true
			}
		}
		if !added {
			words = append(words, bw)
		}
	}

	// lenght of the words (how many words we have in the expr)
	wordsLen := len(words)
	for sampleID, keys := range m.keys {
		for _, key := range keys {
			for wordID, word := range words {
				if wordID == 0 { // {} ...
					if wordID == wordsLen-1 { // {}
						scores[sampleID]++
					} else { // {} ...
						if words[wordID+1] == key.right { // {} x -> x == key.right
							scores[sampleID]++
							wi := strings.Index(expr, word)
							if wordsMap[sampleID] == nil {
								wordsMap[sampleID] = make(map[string][]int)
							}
							wordsMap[sampleID][key.word] = append(wordsMap[sampleID][key.word], wi, wi+len(word))
						}
					}
				} else { // ... {} ... || ... {}
					if wordID == wordsLen-1 { // ... {}
						if words[wordID-1] == key.left {
							scores[sampleID]++
							wi := strings.Index(expr, word)
							if wordsMap[sampleID] == nil {
								wordsMap[sampleID] = make(map[string][]int)
							}
							wordsMap[sampleID][key.word] = append(wordsMap[sampleID][key.word], wi, len(expr))
						}
					} else { /// ... {} ...
						if words[wordID-1] == key.left { // ... x {} ... -> x == key.left
							scores[sampleID]++
							wi := strings.Index(expr, word)
							if wordsMap[sampleID] == nil {
								wordsMap[sampleID] = make(map[string][]int)
							}
							wordsMap[sampleID][key.word] = append(wordsMap[sampleID][key.word], wi)

							lw := len(wordsMap[sampleID][key.word])
							for j := wordID; j < wordsLen; j++ {
								if words[j] == key.right {
									wordsMap[sampleID][key.word] = append(wordsMap[sampleID][key.word], strings.Index(expr, words[j]))
								}
							}
							if reflect.New(m.tpy).Elem().Field(key.fieldID).Kind() == reflect.String {
								if lw == len(wordsMap[sampleID][key.word]) {
									wordsMap[sampleID][key.word] = append(wordsMap[sampleID][key.word], len(expr))
								}
							} else {
								wordsMap[sampleID][key.word] = append(wordsMap[sampleID][key.word], wordsMap[sampleID][key.word][0]+len(word))
							}
						}
						if words[wordID+1] == key.right { // ... {} x ... -> x == key.right
							scores[sampleID]++
						}
					}
				}
			}
		}
	}
	// select the sample with the highest score
	bestScore := 0
	bestSampleID := -1
	for sid, score := range scores {
		if score > bestScore {
			bestScore = score
			bestSampleID = sid
		}
	}
	return bestSampleID, wordsMap[bestSampleID]
}

func (m *model) fit(expr string) interface{} {
	val := reflect.New(m.tpy).Elem()
	sampleID, keywords := m.selectBestSample(strings.ToLower(expr))
	if sampleID != -1 {
		for _, key := range m.keys[sampleID] {
			if indices, ok := keywords[key.word]; ok {
				switch val.Field(key.fieldID).Kind() {
				case reflect.String:
					val.Field(key.fieldID).SetString(string(expr[indices[0]:indices[1]]))
				case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
					s := string(expr[indices[0]:indices[1]])
					v, _ := strconv.ParseInt(s, 10, 0)
					val.Field(key.fieldID).SetInt(v)
				}
			}
		}
	}
	return val.Interface()
}

// Classifier is a text classifier
type Classifier struct {
	naive   *text.NaiveBayes
	classes []*class
	// Output contains the training output for the
	// NaiveBayes algorithm
	Output *bytes.Buffer
}

type class struct {
	name    string
	samples []string
}

// NewClassifier returns a new classifier
func NewClassifier() *Classifier { return &Classifier{Output: bytes.NewBufferString("")} }

// NewClass creates a classification class
func (cls *Classifier) NewClass(name string, samples []string) error {
	if name == "" {
		return fmt.Errorf("class name can't be empty")
	}
	if len(samples) == 0 {
		return fmt.Errorf("samples can't be nil or empty")
	}
	cls.classes = append(cls.classes, &class{name: name, samples: samples})
	return nil
}

// Learn is the ml process for classification
func (cls *Classifier) Learn() error {
	if len(cls.classes) > 0 {
		stream := make(chan base.TextDatapoint)
		errors := make(chan error)
		cls.naive = text.NewNaiveBayes(stream, uint8(len(cls.classes)), base.OnlyWordsAndNumbers)
		cls.naive.Output = cls.Output
		go cls.naive.OnlineLearn(errors)
		for i := range cls.classes {
			for _, s := range cls.classes[i].samples {
				stream <- base.TextDatapoint{
					X: s,
					Y: uint8(i),
				}
			}
		}
		close(stream)
		for {
			err := <-errors
			if err != nil {
				return fmt.Errorf("error occurred while learning: %s", err)
			}
			// training is done!
			break
		}
		return nil
	}
	return fmt.Errorf("register at least one class before learning")
}

// Classify classifies expr and returns the class name
// which expr belongs to
func (cls *Classifier) Classify(expr string) string { return cls.classes[cls.naive.Predict(expr)].name }
