import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

import java.util.*;
import java.io.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.nio.file.Files;
import java.nio.file.*;
import static java.nio.file.StandardCopyOption.*;



public class Main {
    public static void main(String args[]) throws IOException {
        // creates a StanfordCoreNLP object, with POS tagging, lemmatization, NER, parsing, and coreference resolution
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        //Reading a file
        File captions = new File("/home/vthotigar/Flickr8k/token.txt");
        BufferedReader bf = new BufferedReader(new FileReader(captions));
        String caption;

        //intialiaxiotn
        List<String> keywordList = new ArrayList<>();
        List<String> chairImageList = new ArrayList<>();
        List<String> hpImageList = new ArrayList<>();
        List<String> spImageList = new ArrayList<>();
        keywordList.add("chair");
        keywordList.add("headphone");
        keywordList.add("saxophone");


        //intializing map with 0's
        Map<String, Integer> keywordmap = new HashMap<String, Integer>();
        for (String item : keywordList) {
            keywordmap.put(item, 0);
        }
        while((caption = bf.readLine())!=null) {



            // read some text in the text variable
            // Add your text here!

            // create an empty Annotation just with the given text
            Annotation document = new Annotation(caption);

            // run all Annotators on this text
            pipeline.annotate(document);

            // these are all the sentences in this document
            // a CoreMap is essentially a Map that uses class objects as keys and has values with custom types
            List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);

            for (CoreMap sentence : sentences) {
                // traversing the words in the current sentence
                // a CoreLabel is a CoreMap with additional token-specific methods
                for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {

                    // this is the text of the token
                    String word = token.get(CoreAnnotations.TextAnnotation.class);


                    // this is the POS tag of the token
                    String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);


                    if (keywordmap.containsKey(lemma)) {
                        System.out.println(caption);
                        if(lemma.equals("chair")){

                            Pattern pattern = Pattern.compile("[0-9a-z_]*.jpg");
                            Matcher matcher = pattern.matcher(caption);
                            if (matcher.find()){
                                chairImageList.add(matcher.group(0));
                                Path temp = Files.copy(Paths.get("/home/vthotigar/Flickr8k/Flicker8k_Dataset/" + matcher.group(0)), Paths.get("/home/vthotigar/Flickr8k/data/chair/" + matcher.group(0)), REPLACE_EXISTING);
                            }
                        }
                        if(lemma.equals("headphone")){

                            Pattern pattern = Pattern.compile("[0-9a-z_]*.jpg");
                            Matcher matcher = pattern.matcher(caption);
                            if (matcher.find()){
                                hpImageList.add(matcher.group(0));
                                Path temp = Files.copy(Paths.get("/home/vthotigar/Flickr8k/Flicker8k_Dataset/" + matcher.group(0)), Paths.get("/home/vthotigar/Flickr8k/data/headphone/" + matcher.group(0)), REPLACE_EXISTING);


                            }
                        }
                        if(lemma.equals("saxophone")){

                            Pattern pattern = Pattern.compile("[0-9a-z_]*.jpg");
                            Matcher matcher = pattern.matcher(caption);
                            if (matcher.find()){
                                spImageList.add(matcher.group(0));
                                Path temp = Files.copy(Paths.get("/home/vthotigar/Flickr8k/Flicker8k_Dataset/" + matcher.group(0)), Paths.get("/home/vthotigar/Flickr8k/data/saxophone/" + matcher.group(0)), REPLACE_EXISTING);

                            }
                        }
                        keywordmap.put(lemma, keywordmap.get(lemma) + 1);

                    }

                }
            }
        }
        System.out.println("Image Statistics: ");
        System.out.println(keywordmap);
        System.out.println(chairImageList);
        System.out.println(hpImageList);
        System.out.println(spImageList);
    }
}