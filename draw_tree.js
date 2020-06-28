
// Import decision tree from scikit-learn (json file)
var file_name="projet_class.json"
console.log("Hello")

// Set the dimensions and margins of the diagram
var margin = {top: 20, right: 100, bottom: 30, left: 200},
    width = 10000 - margin.left - margin.right,
    height = 1000 - margin.top - margin.bottom;

d3.json(file_name, function(error, data) {
          console.log(data)

    total_size = data.size

    var svg = d3.select("body").append("svg")
                                .attr("width", width + margin.right + margin.left)
                                .attr("height", height + margin.top + margin.bottom)
                                .append("g")
                                .attr("transform", "translate("+ margin.left + "," + margin.top + ")");

    var i = 0,
        duration = 750,
        root;

    // Declares a tree layout and assigns the size
    var tree = d3.tree().size([height, width]);

    // Assigns parent, children, height, depth
    root = d3.hierarchy(data, function(d) { return d.children; });
    root.x0 = height / 2;
    root.y0 = 0;

    // Collapse after the second level
    root.children.forEach(collapse);

    update(root);

    // Collapse the node and all it's children
    function collapse(d) {
    if(d.children) {
        d._children = d.children
        d._children.forEach(collapse)
        d.children = null
    }
    }

    function update(source) {

    // Assigns the x and y position for the nodes
    var data = tree(root);

    // Compute the new tree layout.
    var nodes = data.descendants(),
        links = data.descendants().slice(1);

    // Normalize for fixed-depth.
    nodes.forEach(function(d){ d.y = d.depth * 180});

    // **************************************************************************************************
    //                                      Nodes section                                               *
    // **************************************************************************************************

    // Update the nodes
    var node = svg.selectAll('g.node')
        .data(nodes, function(d) {return d.id || (d.id = ++i); });

    var nodeEnter = node.enter().append('g')
                                    .attr('class', 'node')
                                    .attr("transform", function(d) {
                                        return "translate(" + source.y0 + "," + source.x0 + ")";})
                                    .on('click', click);

    // Add Circle for the nodes
    var circle = nodeEnter.append('circle')
                            .attr('class', 'node')
                            .attr('r', 0.5)
                            .style("fill", function(d) {
                                return d._children ? "lightsteelblue" : "#fff"; });

    // Add rect for the nodes 
    var rectHeight = 20, rectWidth = 120;
    
    rect_name = nodeEnter.append("rect")
                            .attr("width", rectWidth)
                            .attr("height", rectHeight)
                            .attr("x",function(d) {
                                return d.children || d._children ? -13 : 13;})
                            .attr("y",rectHeight*-3)
                            .style("stroke", function(d) { return d.type === "split" ? "#ffeded" : "#ffeded";})
                            .style("fill", function(d) { return d._children ? "#fffafa" : "#ffffff"; });
    
    rect_percent_0 = nodeEnter.append("rect")
                            .attr("width", (rectWidth/2)-5)
                            .attr("height", rectHeight)
                            .attr("x",function(d) {
                                return d.children || d._children ? -13 : 13})
                            .attr("y",rectHeight*-4.5)
                            .style("stroke", function(d) { return d.type === "split" ? "#ffeded" : "#ffeded";})
                            .style("fill", function(d) { return d._children ? "#fffafa" : "#ffffff"; });


    rect_percent_1 = nodeEnter.append("rect")
                            .attr("width", (rectWidth/2)-5)
                            .attr("height", rectHeight)
                            .attr("x",function(d) {
                                return d.children || d._children ? -13 + (rectWidth/2)+5 : 13 + (rectWidth/2)+5;})
                            .attr("y",rectHeight*-4.5)
                            .style("stroke", function(d) { return d.type === "split" ? "#ffeded" : "#ffeded";})
                            .style("fill", function(d) { return d._children ? "#fffafa" : "#ffffff"; });
    
    

     function getTextWidth(text, fontSize, fontFace) {
                            var a = document.createElement('canvas');
                            var b = a.getContext('2d');
                            b.font = fontSize + 'px ' + fontFace;
                            return b.measureText(text).width;
                        } 

    // Add labels for the nodes
    text_name = nodeEnter.append('text')
                        .attr("y", rectHeight*-2.3)
                        .attr("x", function(d) {return -13 + ((rectWidth - getTextWidth(d.data.name))/2)})
                        .style("font-family","Judah")
                        .style("font-size","12px")
                        .attr("dy", ".35em")
                        .attr("text-anchor", "start" )
                        .text(function(d) {
                            return d.data.name; }
                        )

    text_percent_0 = nodeEnter.append("text")
                        .attr("x",function(d) {return -13 + ((rectWidth - getTextWidth(d.data.name))/2) - 7})
                        .attr("y", rectHeight*-4)
                        .attr("dy", ".35em")
                        .attr("text-anchor", "middle" )
                        .style("font-size","10px")
                        .style("font-family","Judah")
                        .style("stroke","#b6232a")
                        .style("stroke-width","0.05em")
                        .text(function(d) {
                            const [v1,,v2] = d.data.pred.match(/\d+/g).map(v => parseInt(v))
                            return Math.round(v1 / (v1 + v2)*100,3) + '%'
                        })

                        function getTextWidth(text, fontSize, fontFace) {
                            var a = document.createElement('canvas');
                            var b = a.getContext('2d');
                            b.font = fontSize + 'px ' + fontFace;
                            return b.measureText(text).width}
                        console.log(getTextWidth('/'))
    
    text_percent_1 = nodeEnter.append("text")
                        .attr("x",function(d) {return (-13 + (rectWidth/2)+5) + ((rectWidth - getTextWidth(d.data.name))/2) - 8;})
                        .attr("y", rectHeight*-4)
                        .attr("dy", ".35em")
                        .attr("text-anchor", "start" )
                        .style("font-size","10px")
                        .style("stroke","#b6232a")
                        .style("stroke-width","0.05em")
                        .text(function(d) {
                            const [v1,,v2] = d.data.pred.match(/\d+/g).map(v => parseInt(v))
                            return Math.round(v2 / (v1 + v2)*100,3) + '%'   
                        }
                        )
                        


    text_total_percent = nodeEnter.append("text")
                        .attr("x",function(d) {return (-13 + (rectWidth/2)+5)  ;})
                        .attr("y", (rectHeight*-4.5) - 10)
                        .attr("dy", ".35em")
                        .attr("text-anchor", "start" )
                        .style("font-size","7px")
                        .style("stroke-width","0.05em")
                        .text(function(d) {  return Math.round(d.data.size / total_size*100) + '%'; })
    
    text_size = nodeEnter.append("text")
                        .attr("x",function(d) {return d.children || d._children ? (-13 + (rectWidth/2) +3) : 13 + (rectWidth/2);})
                        .attr("y", (rectHeight*-4.5) - 10)
                        .attr("dy", ".35em")
                        .attr("text-anchor", "end" )
                        .style("font-size","7px")
                        .style("stroke-width","0.05em")
                        .text(function(d) {  return d.data.size + " /" ; })
    
    text_path_label = nodeEnter.append("text")
                        .attr("x",-13)
                        .attr("y", rectHeight)
                        .attr("dy", ".35em")
                        .attr("text-anchor", "end" )
                        .style("font-size","7px")
                        .text(function(d) {  return d.data.side_bool 
                            ; })                    

 


    // update nodes
    var nodeUpdate = nodeEnter.merge(node);

    nodeUpdate.transition()
                .duration(duration)
                .attr("transform", function(d) { 
                    return "translate(" + d.y + "," + d.x + ")";
                });

    // Update the node attributes and style
    nodeUpdate.select('circle.node')
                .attr('r', 3)
                .style("fill", function(d) {
                    return d._children ? "lightsteelblue" : "#fff";
                })
                .attr('cursor', 'pointer');


    // Remove any exiting nodes
    var nodeExit = node.exit().transition()
                        .duration(duration)
                        .attr("transform", function(d) {
                            return "translate(" + source.y + "," + source.x + ")";
                        })
                        .remove();

    nodeExit.select('circle')
              .attr('r', 1e-6);

    // On exit reduce the opacity of text labels
    nodeExit.select('text')
        .style('fill-opacity', 1e-6);

     // **************************************************************************************************
    //                                      Links section                                               *
    // **************************************************************************************************

    // Update the links.
    var link = svg.selectAll('path.link')
        .data(links, function(d) { return d.id; });

    // Enter any new links at the parent's previous position.
    var linkEnter = link.enter().insert('path', "g")
        .attr("class", "link")
        .attr('d', function(d){
            var o = {x: source.x0, y: source.y0}
            return diagonal(o, o)
        });

    // UPDATE
    var linkUpdate = linkEnter.merge(link);

    // Transition back to the parent element position
    linkUpdate.transition()
        .duration(duration)
        .attr('d', function(d){ return diagonal(d, d.parent) });

    // Remove any exiting links
    var linkExit = link.exit().transition()
        .duration(duration)
        .attr('d', function(d) {
            var o = {x: source.x, y: source.y}
            return diagonal(o, o)
        })
        .remove();

    // Store the old positions for transition.
    nodes.forEach(function(d){
        d.x0 = d.x;
        d.y0 = d.y;
    });

    // Creates a curved (diagonal) path from parent to the child nodes
    function diagonal(s, d) {

        path = `M ${s.y} ${s.x}
                C ${(s.y + d.y) / 2} ${s.x},
                ${(s.y + d.y) / 2} ${d.x},
                ${d.y} ${d.x}`

        return path
    }

    // Toggle children on click.
    function click(d) {
        if (d.children) {
            d._children = d.children;
            d.children = null;
        } else {
            d.children = d._children;
            d._children = null;
        }
        update(d);

  

  }

 
}
})