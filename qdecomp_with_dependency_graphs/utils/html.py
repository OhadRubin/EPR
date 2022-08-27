from typing import Tuple, Callable, List


class HTMLTemplate:
    def __init__(self, styles: List[str] = [], scripts: List[str] = []):
        self.styles: List[str] = styles
        self.scripts: List[str] = scripts

    def get_body(self, question_id: str, body: str):
        styles = '\n'.join(self.styles)
        scripts = '\n'.join(self.scripts)
        return f"""
            <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <title>{question_id}</title>
                    <style>
                    {styles}
                    </style>
                </head>
                <body>
                <p>Question ID: {question_id}</p>
                {body}
                <script>
                {scripts}
                </script>
        """


class CollapsibleHTML(HTMLTemplate):
    style = """
    /* Style the button that is used to open and close the collapsible content */
.collapsible {
  background-color: #eee;
  color: #444;
  cursor: pointer;
  padding: 18px;
  //width: 100%;
  border: none;
  text-align: left;
  outline: none;
  font-size: 15px;
}

/* Add a background color to the button if it is clicked on (add the .active class with JS), and when you move the mouse over it (hover) */
.active, .collapsible:hover {
  background-color: #ccc;
}

/* Style the collapsible content. Note: hidden by default */
.content {
  padding: 0 18px 0 100px;
  display: none;
  //overflow: hidden;
  //background-color: #f1f1f1;
}
    """

    script="""
var coll = document.getElementsByClassName("collapsible");
var i;

for (i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var content = this.nextElementSibling;
    if (content.style.display === "block") {
      content.style.display = "none";
    } else {
      content.style.display = "block";
    }
  });
}
    """
    def __init__(self):
        super().__init__(
            styles=[CollapsibleHTML.style],
            scripts=[CollapsibleHTML.script]
        )

    def wrap_collapsible(self, hint:str, element:str):
        return f"""
            <button type="button" class="collapsible">{hint}</button>
            <div class="content">
              {element}
            </div>
        """



class ToggleBarHTML(HTMLTemplate):
    style = """
        /* Style the button that is used to open and close the collapsible content */
.collapsible {
  background-color: #eee;
  color: #444;
  cursor: pointer;
  padding: 18px;
  //width: 100%;
  border: none;
  text-align: left;
  outline: none;
  font-size: 15px;
}

/* Add a background color to the button if it is clicked on (add the .active class with JS), and when you move the mouse over it (hover) */
.active, .collapsible:hover {
  background-color: #ccc;
}
    """

    script="""
var coll = document.getElementsByClassName("collapsible");
var i;

for (i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    c = this.getAttribute('barToggleClass');
    if (!c){return;}
    this.classList.toggle("active");
    var contents = document.getElementsByClassName(c);
    var is_show = this.classList.contains("active");
    for (j = 0; j < contents.length; j++) {
        var content = contents[j];
        if (is_show) {
          content.style.display = "block";
        } else {
          content.style.display = "none";
        }
    }
  });
}
    """
    def __init__(self, class_names: List[str]):
        super().__init__(
            styles=[ToggleBarHTML.style],
            scripts=[ToggleBarHTML.script]
        )
        self.class_names: List[str] = class_names

    def get_body(self, question_id: str, body: str):
        buttons = '\n'.join(
            f"""<button type="button" class="collapsible active" barToggleClass="{group}">{group}</button>"""
            for group in self.class_names
        )
        new_body = f"""
        {buttons}
        {body}
        """
        return super().get_body(question_id=question_id, body=new_body)



