# Assignment2 CHIARA VENTURI matr. VR504017
For this assignment I choose to solve the first case.
The project focuses on developing a text processing system capable of handling large inputs that exceeds an LLM max input size, using Python.
The code begins by defining the maximum length of the text context window and the size of the overlap for processing text segments.
•	preprocess_text_basic function created for basic text preprocessing, this includes converting the text to lowercase, removing punctuation, tokenizing, lemmatizing, and removing stopwords.
•	process_text_basic  then the system evaluates the size of the input text, if it is within the standard size of the context window, it is processed directly without modification. For inputs exceeding this size, the text is divided into smaller segments or slices (split_text). This is achieved while maintaining an overlap between segments to ensure no loss of context, and each slice is sized to fit within the context window, in fact the total number of slices is determined such that their combined length is greater than or equal to the original input length.
•	cosine_distance  it calculates the cosine similarity between segments, the ones with a similarity below a certain threshold (set at 0.2) are considered distinct enough to be kept separate.
•	read_text_from_pdf_url  process for downloading, reading, and analyzing text from a PDF file
