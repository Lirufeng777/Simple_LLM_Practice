备注：
    transformer.py 为未经封装三层单头注意力机制实现
    improved_transformer.py 为封装的三层单头注意力机制实现
    
    Tokenizer.py 为字符串输入的未封装的分词器实现
    improved_Tokenizer.py 仅封装了类Tokenizer
    
    Tokenizer_baby_demo.py 为特定字符串输入的未封装的写给my only baby cjy的玩具demo
    improved_baby_demo.py 使用了封装的tokenizer和transformer,后面的LM-head、Loss和Train并未写入，属于半成品。