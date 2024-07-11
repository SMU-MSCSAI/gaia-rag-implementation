class TextChunker:
    def __init__(self, chunk_size=512, chunk_overlap=50, separator=" "):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk_text(self, text):
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]

            # Ensure that we do not split the chunk in the middle of a word by finding the last occurrence of the separator
            if end < text_length:
                separator_index = chunk.rfind(self.separator)
                if separator_index != -1:
                    end = start + separator_index + 1  # Adjust the end to include the separator
                    chunk = text[start:end]

            chunks.append(chunk.strip())
            start += self.chunk_size - self.chunk_overlap

        return chunks
