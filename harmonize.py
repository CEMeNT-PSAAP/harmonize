import numpy as np
from os.path import getmtime, exists, dirname, abspath
from os      import makedirs, getcwd
from numba import njit, cuda
import numba
import re
import subprocess
from llvmlite import binding

import inspect
import sys


HARMONIZE_ROOT_DIR =  dirname(abspath(__file__))
HARMONIZE_ROOT_CPP = HARMONIZE_ROOT_DIR+"/harmonize.cpp"


# Uses nvidia-smi to query the compute level of the GPUs on the system. This
# compute level is what is used for compling PTX.
def native_cuda_compute_level():
    query_cmd = "nvidia-smi --query-gpu compute_cap --format=csv,noheader"
    completed = subprocess.run(query_cmd.split(),shell=False,check=True, capture_output=True)
    output = completed.stdout.decode("utf-8").strip().replace(".","")
    return output



DEBUG = False

# Injects `value` as the value of the global variable named `name` in the module
# that defined the function `index` calls down the stack, from the perspective
# of the function that calls `inject_global`.
def inject_global(name,value,index):
    frm = inspect.stack()[index+1]
    mod = inspect.getmodule(frm[0])
    setattr(sys.modules[mod.__name__], name, value)
    print(f"Defined '{name}' as "+str(value)+" for module "+mod.__name__)


# Returns the type annotations of the arguments for the input function
def fn_arg_ano_list( func ):
    result = []
    for arg,ano in func.__annotations__.items():
        if( arg != 'return' ):
            result.append(ano)
    return result

# Returns the numba type of the function signature of the input function
def fn_sig( func ):
    arg_list = []
    ret    = numba.types.void
    for arg,ano in func.__annotations__.items():
        if( arg != 'return' ):
            arg_list.append(ano)
        else:
            ret = ano
    return ret(*arg_list)


# Returns the type annotations of the arguments for the input function
# as a tuple
def fn_arg_ano( func ):
    return tuple( x for x in fn_arg_ano_list(func) )


# Raises an error if the annotated return type for the input function
# does not patch the input result type
def assert_fn_res_ano( func, res_type ):
    if 'return' in func.__annotations__:
        res_ano = func.__annotations__['return']
        if res_ano != res_type :
            arg_str  = str(fn_arg_ano(func))
            ano_str  = arg_str + " -> " + str(res_ano)
            cmp_str  = arg_str + " -> " + str(res_type)
            err_str  = "Annotated function type '" + ano_str               \
            + "' does not match the type deduced by the compiler '"        \
            + cmp_str + "'\nMake sure the definition of the function '"    \
            + func.__name__ + "' results in a return type  matching its "  \
            + "annotation when supplied arguments matching its annotation."
            raise(TypeError(err_str))


# Returns the ptx of the input function, as a global CUDA function. If the return type deduced
# by compilation is not consistent with the annotated return type, an exception is raised.
def global_ptx( func ):
    ptx, res_type = cuda.compile_ptx_for_current_device(func,fn_arg_ano(func),debug=DEBUG,opt=(not DEBUG))
    assert_fn_res_ano(func, res_type)
    return ptx

# Returns the ptx of the input function, as a device CUDA function. If the return type deduced
# by compilation is not consistent with the annotated return type, an exception is raised.
def device_ptx( func ):
    print(func.__name__)
    ptx, res_type = cuda.compile_ptx_for_current_device(func,fn_arg_ano(func),device=True,debug=DEBUG,opt=(not DEBUG))
    assert_fn_res_ano(func, res_type)
    return ptx, res_type

# Returns the modify date of the file containing the supplied function. This is useful for
# detecting if a function was possibly changed
def func_defn_time(func):
    return getmtime(func.__globals__['__file__'])



# Removes all comments from the input ptx, to aid its subsequent parsing, returning the
# sanitized result string
def remove_ptx_comments(ptx_text):
    space = ' \t\r\n'
    ptx_text = "\n".join([ line.lstrip(space) for line in ptx_text.splitlines() if len(line.lstrip(space)) != 0 ])

    filtered = []
    start_split = ptx_text.split("/*")
    for index, entry in enumerate(start_split) :
        end_split = entry.split("*/")
        if index == 0 :
            if len(end_split) != 1 :
                raise AssertionError("PTX has an unmatched comment block end")
            filtered.append(end_split[0])
        elif len(end_split) !=2 :
            raise AssertionError("PTX has an unmatched comment block end")
        else :
            filtered.append(end_split[1])

    ptx_text = "".join(filtered)

    ptx_text = "\n".join([ line.lstrip(space) for line in ptx_text.splitlines() if not line.lstrip(space).startswith("//") ])

    ptx_text = "\n".join([ line.split("//")[0] for line in ptx_text.splitlines() ])

    return ptx_text

# Parses the input text based upon bracket delimiters (such as (), {}, [], <> ), returning
# the resulting parse tree. Leaf nodes are strings.
def parse_braks(ptx_text,paren_pairs,closer=None):
    seq = []
    limit = len(ptx_text)
    start = 0
    index = 0
    while index < limit :
        character = ptx_text[index]
        if character == closer :
            seq.append(ptx_text[start:index])
            return ( closer, seq ), index + 1
        for open,close in paren_pairs :
            if character == open :
                seq.append(ptx_text[start:index])
                index += 1
                sub_seq, delta = parse_braks(ptx_text[index:limit],paren_pairs,close)
                seq.append(sub_seq)
                index += delta
                start = index
        index += 1
    seq.append(ptx_text[start:index])
    return ( closer, seq ), index



# Recursively iterates through a bracket-delimiter parse tree, breaking/grouping nodes by
# the delimiters contained within the leaf nodes
def parse_sep(ptx_brak_tree, sep):
    closer, seq = ptx_brak_tree
    new_seq = []
    sep_found = False
    sep_seq = []
    seq_len = len(seq)
    for idx, sub_seq in enumerate(seq) :
        if isinstance(sub_seq,str) :
            split_seq = sub_seq.split(sep)
            length = len(split_seq)
            if length == 1 :
                sep_seq.append(split_seq[0])
            elif length > 1 :
                sep_found = True
                sep_seq.append(split_seq[0])
                new_seq.append((sep,sep_seq))
                sep_seq = [split_seq[-1]]
                for split in split_seq[1:-1]:
                    new_seq.append((sep,[split]))
        elif sub_seq[0] == '}':
            capped = False
            not_last = (idx < seq_len -1)
            incomplete = False
            if len(sep_seq) > 0 and isinstance(sep_seq[0],str):
                incomplete = not sep_seq[0].isspace()
            if incomplete and not_last:
                next = seq[idx+1]
                if isinstance(next,str):
                    split = next.split(sep)
                    empty = len(split[0]) == 0 or split[0].isspace()
                    if len(split) > 1 and empty:
                        capped = True
            if incomplete and not_last and capped:
                sep_seq.append(sub_seq)
                new_seq.append((sep,sep_seq))
                sep_seq = []
            else:
                new_seq += sep_seq
                sep_seq = []
                new_seq.append(parse_sep(sub_seq,sep))
        else :
            sep_seq.append(parse_sep(sub_seq,sep))
    if sep_found :
        new_seq.append((sep,sep_seq))
        return (closer,new_seq)
    else :
        new_seq += sep_seq
        return (closer,new_seq)


# Breaks up the leaf nodes of a parse tree by whitespace
def parse_tok(parse_tree):
    closer, seq = parse_tree
    new_seq = []
    for chunk in seq:
        if isinstance(chunk,str):
            sub_seq = chunk.split()
            if len(sub_seq) > 0:
                new_seq.append((' ',sub_seq))
        else:
            sub_tree = parse_tok(chunk)
            if sub_tree[0] != ' ' or len(sub_tree[1]) != 0:
                new_seq.append(parse_tok(chunk))
    return (closer,new_seq)


# Parses ptx into an AST based upon bracket delimiters and seperators
def parse_ptx(ptx_text):
    ptx_text = remove_ptx_comments(ptx_text)
    braks = [ ('(',')'), ('[',']'), ('{','}'), ('<','>') ]
    parse_tree, _ = parse_braks(ptx_text,braks)
    seperators  = [ ';' , ',', ':' ]
    for sep in seperators:
        parse_tree = parse_sep(parse_tree,sep)
    parse_tree = parse_tok(parse_tree)
    return parse_tree

# Checks that a sequence of nodes matches the supplied pattern of delimiters
def delim_match(chunk_list,delim_list):
    if len(delim_list) > len(chunk_list):
        return False
    for index, delim in enumerate(delim_list):
        if chunk_list[index][0] != delim:
            return False
    return True



# Searches a parse tree for leaves matching the input regex
def extract_regex(parse_tree,regex):
    result = []
    if isinstance(parse_tree,str):
        #for match in re.finditer(regex,parse_tree):
        #    result.append((match, None, None))
        return result

    _, chunks = parse_tree
    for index, chunk in enumerate(chunks):
        if isinstance(chunk,str):
            continue
        sep, content = chunk
        if sep != "ptx":
            result += extract_regex((sep,content),regex)
            continue
        for match in re.finditer(regex,content):
            result.append((match, index, chunks))
    return result


# Searches the input parse tree for extern functions, returning any matches
def find_extern_funcs(parse_tree):
    result = []
    extern_regex = r'.extern\s+.func\s*\(\s*.param\s\.+\w+\s+\w+\s*\)\s*(?P<name>\w+)\((?P<params>(\s*.param\s+\.\w+\s+\w+\s*)(,\s*.param\s+\.\w+\s+\w+\s*)*)\)'
    param_regex = r'\s*.param\s+(?P<type>\.\w+)\s+(?P<name>\w+)\s*'
    for match, _, _ in extract_regex(parse_tree,extern_regex):
        params = [(p['type'],p['name']) for p in re.finditer(param_regex,match['params'])]
        result.append((match['name'],params))
    return result


# Searches the input parse tree for visible (non-extern) functions, returning any matches
def find_visible_funcs(parse_tree):
    result = []
    extern_regex = r'\.visible\s+\.func\s*\(\s*\.param\s+\.\w+\s+\w+\s*\)\s*(?P<name>\w+)\((?P<params>(\s*\.param\s+\.\w+\s+\w+\s*)(,\s*.param\s+\.\w+\s+\w+\s*)*)\)'
    param_regex = r'\s*\.param\s+(?P<type>\.\w+)\s+(?P<name>\w+)\s*'
    for match, index, context in extract_regex(parse_tree,extern_regex):
        params = [(p['type'],p['name']) for p in re.finditer(param_regex,match['params'])]
        result.append((match['name'],params,context[index+1]))
    return result


# Replaces leaf nodes (or portions of leaf nodes) based off of a list
# of 'target' and 'destination' patterns. If a target pattern is found, it
# is replaced with its corresponding destination.
def replace(parse_tree,rep_list,whole=False):
    if isinstance(parse_tree,str):
        for targ, dest in rep_list:
            pattern = targ
            if whole:
                pattern = "\\b" + pattern + "\\b"
            parse_tree = re.sub(pattern,dest,parse_tree)
        return parse_tree
    else:
        sep, content = parse_tree
        if sep == "ptx":
            return (sep, replace(content,rep_list,whole))
        new_content = []
        for chunk in content:
            new_content.append(replace(chunk,rep_list,whole))
        return (sep,new_content)


# Returns true if and only if the input parse tree has a curly-brace block
def has_curly_block(parse_tree):
    if isinstance(parse_tree,str):
        return False
    sep, content = parse_tree
    if sep == '}':
        return True
    for chunk in content:
        if has_curly_block(chunk):
            return True
    return False

# Returns true if and only if the input parse tree has a colon
def has_colon(parse_tree):
    if isinstance(parse_tree,str):
        return False
    sep, content = parse_tree
    if sep == ':':
        return True
    for chunk in content:
        if has_colon(chunk):
            return True
    return False

# Collapses the portions of a parse tree that correspond to the same line in
# ptx syntax. This is useful for performing whole-line regex matches.
def linify_tree(parse_tree):
    if isinstance(parse_tree,str):
        return parse_tree
    sep, content = parse_tree
    if sep ==';': # and not has_curly_block(parse_tree):
        return  ("ptx",stringify_tree(parse_tree).replace("\n","")+"\n")

    hit_curly   = False
    hit_line    = False
    line        = []
    new_content = []
    for chunk in content:
        if isinstance(chunk,str):
            new_content.append(chunk)
            continue
        chunk = linify_tree(chunk)
        sub_sep, sub_con = chunk
        if sub_sep == "ptx":
            hit_line = True
            split = sub_con.split(':')
            for index, sub_chunk in enumerate(split):
                #if index != 0:
                #    sub_chunk = sub_chunk[1:]
                if index != len(split)-1 :
                    new_content.append(("ptx",sub_chunk+":\n"))
                else:
                    new_content.append(("ptx",sub_chunk))
        elif sub_sep == '}':
            hit_curly = True
            new_content.append(("ptx",stringify_tree((None,line))+"\n"))
            line = []
            new_content.append((sub_sep,sub_con))
        else:
            line.append((sub_sep,sub_con))
    if len(line) != 0:
        if (hit_curly or hit_line):
            new_content.append(("ptx",stringify_tree((None,line))+"\n"))
        else:
            new_content += line
    return (sep,new_content)

# Converts a parse tree into its equivalent text, returned as a string
def stringify_tree(parse_tree,inlined=False,last=False,depth=0):
    brak_list = [')',']','}','>']
    brak_map  = {')':'(',']':'[','}':'{','>':'<'}
    sep_list  = [',',';',':']

    tabs = '\t'*depth

    if isinstance(parse_tree,str):
        return " " + parse_tree
    else:
        sep, content = parse_tree

        if sep == "cuda":
            return tabs + content

        if sep == "ptx":
            if content.strip() == ";":
                return tabs + '\n' #tabs + ';' + '\n'
            elif inlined:
                return tabs + "asm volatile (\"" + content.rstrip() + "\" ::: \"memory\");\n"
            else:
                return tabs + content

        if sep == ' ':
            return "\t".join(content)

        sub_depth = depth

        result = ""

        if sep in brak_list:
            if sep == '}':
                sub_depth += 1
                if inlined:
                    result += tabs + "//asm volatile (\"" + str(brak_map[sep]) + "\");\n"
                else:
                    result += tabs + str(brak_map[sep]) + '\n'
            else:
                result += str(brak_map[sep])

        for index, chunk in enumerate(content):
            sub_last = (index == (len(content)-1))
            result += stringify_tree(chunk,inlined,sub_last,sub_depth)

        if sep in brak_list:
            if sep == '}':
                if inlined:
                    result += tabs + "//asm volatile (\"" + str(sep) + "\");\n"
                else:
                    result += tabs + str(sep) + '\n'
            else:
                result += str(sep)

        elif (sep in sep_list) and not last:
            result += str(sep)
            if sep == ';' or sep == ':':
                result += '\n' + tabs
            else:
                result += '\t'
        return result




# Removes versioning/config information and comments that are not needed in the final ptx
def strip_ptx(ptx_text,inline):
    ignore_list = [ "//" ]
    if inline:
        ignore_list.extend([".version", ".target" , ".address_size", ".common" ])
    # get rid of empty lines and leading whitespace
    space = ' \t\r\n'
    ptx_text = "\n".join([ line.lstrip(space) for line in ptx_text.splitlines() if len(line.lstrip(space)) != 0 ])
    # get rid of comments and other ptx lines we don't care about
    for entry in ignore_list:
        ptx_text = "\n".join([ line for line in ptx_text.splitlines() if not line.lstrip(space).startswith(entry) ])
    return ptx_text



# (Currently unused) Replaces the call parameters of a function body with appropriate move
# instructions. This function has potential use for inlining ptx.
def replace_call_params(context,call_idx,ret,signature,params,temp_count):
    kind_map = {
        "u8":"h", "u16":"h", "u32":"r", "u64":"l",
        "s8":"h", "s16":"h", "s32":"r", "s64":"l",
        "b8":"h", "b16":"h", "b32":"r", "b64":"l",
        "f32":"f", "f64":"d",
    }

    params.append(ret)

    for id, param in enumerate(params):
        decl_regex = r"\.param\s+\.\w+\s+"+param+r"\s*;"
        move_regex = r"st\.param\.(?P<kind>\w+)\s*\[\s*"+param+r"\s*(\+\s*[0-9]+\s*)?\]\s*,\s*(?P<src>\w+)\s*;"
        if param == ret:
            move_regex = r"ld\.param\.(?P<kind>\w+)\s*(?P<dst>\w+)\s*,\s*\[\s*"+param+r"\+0\]\s*;"

        start = min(call_idx+2,len(context))
        found_decl = False
        found_move = False
        for line_idx in range(start,-1,-1):
            if found_decl and found_move:
                break
            if isinstance(context[line_idx],str):
                continue
            if context[line_idx][0] != 'ptx':
                continue
            if re.match(decl_regex,context[line_idx][1]) != None:
                found_decl = True
                if param == ret:
                    context[line_idx] = ('cuda', "//"+context[line_idx][1])
                else:
                    if id == 0:
                        line = '{kind} *_param_{id};\n'
                    else:
                        line = '{kind} _param_{id};\n'
                    line = line.format(kind=str(signature[id]),id=temp_count+id)
                    context[line_idx] = ('cuda',line)
                continue
            move_match = re.match(move_regex,context[line_idx][1])
            if move_match != None:
                found_move = True
                kind = move_match['kind']
                if param == ret:
                    dst = move_match['dst']
                    #line = 'asm volatile(\"cvt.{kind}.{kind} {dst}, 0;\");\n'
                    line = 'asm volatile(\"mov.{kind} {dst}, 0;\" ::: "memory" );\n'
                    line = line.format(kind=kind,dst=dst)
                    context[line_idx] = ('cuda',line)
                else:
                    src  = move_match['src']
                    #line = 'asm volatile(\"cvt.{kind}.{kind} %0, {src};\" : \"={kid}\"(_param_{id}) : );\n'
                    line = 'asm volatile(\"mov.{kind} %0, {src};\" : \"={kid}\"(_param_{id}) :: "memory" );\n'
                    line = line.format(kind=kind,src=src,id=temp_count+id,kid=kind_map[kind])
                    context[line_idx] = ('cuda',line)
    return temp_count + len(params)-1




def replace_call( parse_tree, mapping, temp_count, type_map, context=[] ):
    repl_fn, repl_name = mapping
    src_list = []
    ret  = None
    call_regex = r"call\.uni\s*\(\s*(?P<ret>\w+)\s*\)\s*,\s*"+repl_fn.name+r"\s*,\s*\((?P<params>\s*\w+\s*(,\s*\w+\s*)*)\)\s*;\s*"
    param_regex = r'\b(?P<name>\w+)\b'
    for match, index, context in extract_regex(parse_tree,call_regex):

        ret         = match['ret']
        params      = [p['name'] for p in re.finditer(param_regex,match['params'])]
        signature   = [ repl_fn.sig.return_type ] + [ arg for arg in repl_fn.sig.args ]
        signature   = [ map_type_name(type_map,kind) for kind in signature ]

        old_tc = temp_count
        temp_count = replace_call_params(context,index,ret,signature,params,temp_count)
        arg_str = ",".join([ "_param_"+str(x) for x in range(old_tc+1,temp_count)])

        if signature[0] == "void":
            context[index] = ('cuda', repl_name + "(" + arg_str + ");\n" )
        else:
            context[index] = ('cuda', "*_param_0 = " + repl_name + "(" + arg_str + ");\n" )

    return temp_count




def replace_externs( parse_tree, function_map, type_map ):
    temp_count = 0
    for mapping in function_map.items():
        temp_count = replace_call(parse_tree,mapping,temp_count,type_map)



def replace_fn_params( parse_tree, params, has_return ):
    kind_map = {
        "u8":"h", "u16":"h", "u32":"r", "u64":"l",
        "s8":"h", "s16":"h", "s32":"r", "s64":"l",
        "b8":"h", "b16":"h", "b32":"r", "b64":"l",
                             "f32":"f", "f64":"d",
    }
    param_regex = r"ld\.param\.(?P<kind>\w+)\s+(?P<dst>\w+)\s*,\s*\[(?P<name>\w+)\]\s*;"

    for match, index, context in extract_regex(parse_tree,param_regex):
        kind = match['kind']
        dst  = match['dst']
        name = match['name']

        if name not in params:
            continue

        line = 'asm volatile (\"cvt.{kind}.{kind} {dst}, %0;\" : : \"{kid}\"({name}) : "memory" );\n'
        line = line.format(kind=kind,dst=dst,name=name,kid=kind_map[kind])
        context[index] = ('cuda',line)

    return_regex = r"st\.param\.(?P<kind>\w+)\s*\[func_retval0\+0\]\s*,\s*(?P<src>\w+)\s*;"
    for match, index, context in extract_regex(parse_tree,return_regex):
        #if has_return:
        #    kind = match['kind']
        #    src  = match['src']
        #    line = 'asm volatile (\"cvt.{kind}.{kind} %0, {src}\" : \"={kid}\"(result) : );\n'
        #    line = line.format(kind=kind,src=src,kid=kind_map[kind])
        #    context[index] = ('cuda',line )
        #else:
        context[index] = ('cuda',"//"+context[index][1])



def fix_returns(parse_tree,fn_name):
    return_regex = r"ret\s*;"
    for _, index, context in extract_regex(parse_tree,return_regex):
        context[index] = ('ptx',"bra $RETURN_"+fn_name+";")


def fix_temp_param(parse_tree):
    return_regex = r"temp_param_reg;"
    for _, index, context in extract_regex(parse_tree,return_regex):
        context[index] = ('cuda',"//"+context[index][1])



def extern_device_ptx( func, type_map ):
    arg_types   = fn_arg_ano_list(func)
    ptx_text, res_type = device_ptx(func)
    ptx_text = strip_ptx(ptx_text,False)
    parse_tree = parse_ptx(ptx_text)
    line_tree  = linify_tree(parse_tree)
    extern_regex = r'(?P<before>\.visible\s+\.func\s*\(\s*\.param\s+\.\w+\s+\w+\s*\)\s*)(?P<name>\w+)(?P<after>\((?P<params>(\s*\.param\s+\.\w+\s+\w+\s*)(,\s*.param\s+\.\w+\s+\w+\s*)*)\))'
    for match, index, context in extract_regex(line_tree,extern_regex):
        before = match['before']
        after  = match['after']
        context[index] = before + "_" + func.__name__ + after
    ptx_text = stringify_tree(line_tree,False)
    ptx_text = ptx_text.replace("call.uni","call")
    return ptx_text




def inlined_device_ptx( func, function_map, type_map ):
    arg_types   = fn_arg_ano_list(func)
    ptx_text, res_type = device_ptx(func)
    parse_tree  = parse_ptx(ptx_text)
    line_tree   = linify_tree(parse_tree)
    visible_fns = find_visible_funcs(line_tree)

    if len(visible_fns) == 0:
        raise AssertionError("PTX for user-defined function contains no '.visible .func'")
    elif len(visible_fns) > 1:
        raise AssertionError("PTX for user-defined function contains multiple '.visible .func'")


    name, params, body = visible_fns[0]

    whole_tf_list = []
    inner_tf_list = [
        ("%","_"+func.__name__+"_"),
        ("\$L__","$L_"+func.__name__+"_"),
        ("__local_depot0","_"+func.__name__+"_local_depot0")
    ]
    par_types     = []

    whole_tf_list.append((name,func.__name__))

    targ_par_names  = fn_arg_ano_list(func)

    for index, param in enumerate(params):
        par_name = param[1]
        whole_tf_list.append((par_name,"fn_param_"+str(index)))

    body = replace(body,whole_tf_list,whole=True)
    body = replace(body,inner_tf_list,whole=False)

    replace_externs(body,function_map,type_map)

    has_return = 'return' in func.__annotations__

    replace_fn_params(body,["fn_param_"+str(index) for index in range(len(params))], has_return)
    fix_returns(body,func.__name__)
    fix_temp_param(body)
    body[1].append(('ptx',"$RETURN_"+func.__name__+":"))
    cuda_body = stringify_tree(body,False,False,2)

    return cuda_body


# Used to map numba types to numpy d_types
def map_type_to_np(kind):
    return numba.np.numpy_support.as_dtype(kind)
    primitives = {
        numba.none    : np.void,
        bool          : np.bool8,
        numba.boolean : np.bool8,
        numba.uint8   : np.uint8,
        numba.uint16  : np.uint16,
        numba.uint32  : np.uint32,
        numba.uint64  : np.uint64,
        numba.int8    : np.int8,
        numba.int16   : np.int16,
        numba.int32   : np.int32,
        numba.int64   : np.int64,
        numba.float32 : np.float32,
        numba.float64 : np.float64
    }
    if kind in primitives:
        return primitives[kind]
    return kind


# Determines the alignment of an input record type. For proper operation,
# the input type MUST be a record type
def alignment(kind):
    align = 1
    for name,type in kind.members:
        member_align = kind.alignof(name)
        if member_align != None and member_align > align:
            align = member_align
    return align


# Maps an input type to the CUDA/C++ equivalent type name used by Harmonize
def map_type_name(type_map,kind,rec_mode=""):
    primitives = {
        numba.none : "void",
        bool       : "bool",
        np.bool8   : "bool",
        np.uint8   : "uint8_t",
        np.uint16  : "uint16_t",
        np.uint32  : "uint32_t",
        np.uint64  : "uint64_t",
        np.int8    : "int8_t",
        np.int16   : "int16_t",
        np.int32   : "int32_t",
        np.int64   : "int64_t",
        np.float32 : "float",
        np.float64 : "double"
    }


    if kind in primitives:
        return primitives[kind]
    elif isinstance(kind,numba.types.abstract.Literal):
        return map_type_name(type_map,type(kind._literal_value))
    elif isinstance(kind,numba.types.Integer):
        result = "int"
        if not kind.signed :
            result = "u" + result
        result += str(kind.bitwidth)
        return result + "_t"
    elif isinstance(kind,numba.types.Float):
        return "float" + str(kind.bitwidth)
    elif isinstance(kind,numba.types.Boolean):
        return "bool"
    elif isinstance(kind,numba.types.Record):
        #if kind in type_map:
        #    return type_map[kind] + "*"
        #else:
        size  = kind.size
        align = alignment(kind)
        align = 8
        size  = ((size + (align-1)) // align) * align
        result = "_"+str(size)+"b"+str(align)
        if rec_mode == "ptr":
            result += "*"
        elif rec_mode == "void_ptr":
            result = "void*"
        return result
    elif isinstance(kind,numba.types.npytypes.NestedArray):
        return "void*"
    else:
        raise RuntimeError("Unrecognized type '"+str(kind)+"' with type '"+str(type(kind))+"'")


# Returns the number of arguments used by the input function
def arg_count(func):
    return len(fn_arg_ano_list(func))


# Returns the CUDA/C++ text used as the parameter list for the input function, with various
# options to map record type names, account for preceding parameters, remove initial
# parameters from the signature, or force the first to be a void*.
def func_param_text(func,type_map,rec_mode="",prior=False,clip=0,first_void=False):
    param_list  = fn_arg_ano_list(func)

    param_list = param_list[clip:]

    if first_void:
        param_list = param_list[1:]
        param_list  = ["void*"] + [ map_type_name(type_map,kind,rec_mode=rec_mode) for kind in param_list ]
    else:
        param_list  = [ map_type_name(type_map,kind,rec_mode=rec_mode) for kind in param_list ]

    param_text  = ", ".join([ kind+" fn_param_"+str(idx+clip+1) for (idx,kind) in enumerate(param_list)])
    if len(param_list) > 0 and prior:
        param_text = ", " + param_text
    return param_text


# Returns the CUDA/C++ text used as the argument list for a call to the input function,
# with various options to cast/deref/getadr values, account for preceding parameters, and
# remove initial parameters from the signature.
def func_arg_text(func,type_map,rec_mode="",prior=False,clip=0):
    param_list  = fn_arg_ano_list(func)
    param_list = param_list[clip:]

    arg_text  = ""
    if len(param_list) > 0 and prior:
        arg_text = ", "
    for (idx,kind) in enumerate(param_list):
        if idx != 0:
            arg_text += ", "
        if isinstance(kind,numba.types.Record):
            if   rec_mode == "deref":
                arg_text += "*"
            elif rec_mode == "adrof":
                arg_text += "&"
            elif rec_mode == "cast_deref":
                arg_text += "*("+map_type_name(type_map,kind,rec_mode="ptr")+")"
        arg_text += "fn_param_"+str(idx+clip+1)
    return arg_text



# Returns the CUDA/C++ text representing the input function as a Harmonize async template
# function. The wrapper type that surrounds such templates is handled in other functions.
def harm_template_func(func,template_name,function_map,type_map,inline,base=False):


    return_type = "void"
    if 'return' in func.__annotations__:
        return_type = map_type_name(type_map, func.__annotations__['return'])


    param_text  = func_param_text(func,type_map,prior=True,clip=1,first_void=False)
    arg_text    = func_arg_text  (func,type_map,rec_mode="adrof",prior=True,clip=1)

    if base:
        param_text = ""
        arg_text   = ""


    code = "\ttemplate<typename PROGRAM>\n"   \
	     + "\t__device__ static "  \
         + return_type+" "+template_name+"(PROGRAM prog" + param_text + ") {\n"

    if return_type != "void":
        code += "\t\t"+return_type+"  result;\n"
        code += "\t\t"+return_type+" *fn_param_0 = &result;\n"
    else:
        code += "\t\tint  dummy_void_result = 0;\n"
        code += "\t\tint *fn_param_0 = &dummy_void_result;\n"


    #code += "\tprintf(\"{"+func.__name__+"}\");\n"
    #code += "\tprintf(\"(prog%p)\",&prog);\n"
    #code += "\tprintf(\"{ctx%p}\",&prog._dev_ctx);\n"
    #code += "\tprintf(\"(sta%p)\",prog.device);\n"
    #code += "\t//printf(\"{pre%p}\",preamble(prog.device));\n"

    if inline:
        code += inlined_device_ptx(func,function_map,type_map)
    else:
        code += "\t\t_"+func.__name__+"(fn_param_0, &prog"+arg_text+");\n"

    if return_type != "void":
        code += "\t\treturn result;\n"
    code += "\t}\n"

    return code


# Returns the input string in pascal case
def pascal_case(name):
    return name.replace("_", " ").title().replace(" ", "")


# Returns the CUDA/C++ text for a Harmonize async function that would map onto the
# input function.
def harm_async_func(func, function_map, type_map,inline):
    return_type = "void"
    if 'return' in func.__annotations__:
        return_type = map_type_name(type_map, func.__annotations__['return'])
    func_name = str(func.__name__)

    struct_name = pascal_case(func_name)
    param_list  = fn_arg_ano_list(func)[1:]
    param_list  = [ map_type_name(type_map,kind) for kind in param_list ]
    param_text  = ", ".join(param_list)

    code = "struct " + struct_name + " {\n"                                              \
	     + "\tusing Type = " + return_type + "(*)(" + param_text + ");\n"                \
         + harm_template_func(func,"eval",function_map,type_map,inline)                  \
         + "};\n"

    return code



# Returns the address of the input device array
def cuda_adr_of(array):
    return array.__cuda_array_interface__['data'][0]






# A base class representing all possible runtime types
class Runtime():
    def __init__(self,spec,context,state,init_fn,exec_fn):
        pass

    def init(self,*args):
        pass

    def exec(self,*args):
        pass

    def load_state(self):
        pass

    def store_state(self):
        pass



# The class representing the default Harmonize runtime type, which
# asynchronously schedules calls on-GPU and within a single kernel
class HarmonizeRuntime(Runtime):

    def __init__(self,spec,context,state,fn,gpu_objects):
        self.spec          = spec
        self.context       = context
        self.context_ptr   = cuda_adr_of(context)
        self.state         = state
        self.state_ptr     = cuda_adr_of(state)
        self.fn            = fn
        self.gpu_objects   = gpu_objects

    # Initializes the runtime using `wg_count` work groups
    def init(self,wg_count=1):
        self.fn["init_program"](self.context_ptr,self.state_ptr,wg_count,32)

    # Executes the runtime through `cycle_count` iterations using `wg_count` work_groups
    def exec(self,cycle_count,wg_count=1):
        self.fn["exec_program"](self.context_ptr,self.state_ptr,cycle_count,wg_count,32)

    # Loads the device state of the runtime, returning value normally and also through the
    # supplied argument, if it is not None
    def load_state(self,result = None):
        res = self.state.copy_to_host(result)
        return res


    # Stores the input value into the device state
    def store_state(self,state):
        cpu_state = np.zeros((1,),self.spec.dev_state)
        cpu_state[0] = state
        return self.state.copy_to_device(cpu_state)


# The Harmonize implementation of standard event-based execution
class EventRuntime(Runtime):


    def __init__(self,spec,context,state,fn,checkout,io_buffers):
        self.spec          = spec
        self.context       = context
        self.context_ptr   = cuda_adr_of(context)
        self.state         = state
        self.state_ptr     = cuda_adr_of(state)
        self.fn            = fn
        self.checkout      = checkout
        self.io_buffers    = io_buffers

    # Used for debugging. Prints out the contents of the buffers used to store
    # intermediate data
    def io_summary(self):
        for field in ["handle","data_a","data_b"]:
            bufs = [ buf[field].copy_to_host() for buf in self.io_buffers ]
            for buf in bufs:
                print(buf)
        return True

    # Returns true if and only if the instance has halted, indicating that no
    # work is left to be performed
    def halted(self):
        bufs = [ buf["handle"].copy_to_host()[0] for buf in self.io_buffers ]
        for buf in bufs:
            if buf["input_iter"]["limit"] != 0:
                return False
        return True

    # Does nothing, since no on-GPU initialization is required for the runtime
    def init(self,wg_count=1):
        pass

    # Executes the program, claiming events from the intermediate data buffers in
    # `chunk_size`-sized units for each thread. Execution continues until the program
    # is halted
    def exec(self,chunk_size,wg_count=1):
        has_halted = False
        count = 0
        while not has_halted:
            self.exec_fn[wg_count,32](self.context_ptr,self.state_ptr,chunk_size)
            has_halted = self.halted()
            count += 1

    # Loads the device state of the runtime, returning value normally and also through the
    # supplied argument, if it is not None
    def load_state(self,result = None):
        res = self.state.copy_to_host(result)
        return res

    # Stores the input value into the device state
    def store_state(self,state):
        cpu_state = np.zeros((1,),self.spec.dev_state)
        cpu_state[0] = state
        return self.state.copy_to_device(cpu_state)


# Represents the specification for a specific program and its
# runtime meta-parameters
class RuntimeSpec():

    def __init__(
            self,
            # The name of the program specification
            spec_name,
            # A triplet containing types which should be
            # held per-device, per-group, and per-thread
            state_spec,
            # A triplet containing the functions that
            # should be run to initialize/finalize states
            # and to generate work for running programs
            base_fns,
            # A list of python functions that are to be
            # included as async functions in the program
            async_fns,
            **kwargs
        ):

        self.spec_name = spec_name


        # The states tracked per-device, per-group, and per-thread
        self.dev_state, self.grp_state, self.thd_state = state_spec

        # The base functions used to set up states and generate work
        self.init_fn, self.final_fn, self.source_fn = base_fns

        self.async_fns = async_fns

        self.meta = kwargs

        if 'function_map' in kwargs:
            self.function_map = kwargs['function_map']
        else:
            self.function_map = {}

        if 'type_map' in kwargs:
            self.type_map = kwargs['type_map']
        else:
            self.type_map = {}

        self.generate_meta()
        self.generate_code()




    # Generates the meta-parameters used by the runtimme, applying defaults for whichever
    # fields that aren't defined
    def generate_meta(self):

        # The number of work links that work groups hold in their shared memory
        if 'STASH_SIZE' not in self.meta:
            self.meta['STASH_SIZE'] = 8

        # The size of the work groups used by the program. Currently, the only
        # supported size is 32
        if 'GROUP_SIZE' not in self.meta:
            self.meta['GROUP_SIZE'] = 32

        # Finding all input types, each represented as dtypes
        input_list = []
        for async_fn in self.async_fns:
            arg_list = []
            for idx, arg in enumerate(fn_arg_ano_list(async_fn)):
                np_type = map_type_to_np(arg)
                arg_list.append(("param_"+str(idx),np_type))
            input_list.append(np.dtype(arg_list))

        # The maximum size of all input types. This must be known in order to
        # allocate memory spans with a size that can support all async functions
        max_input_size = 0
        for input in input_list:
            max_input_size = max(max_input_size,input.itemsize)
        self.meta['MAX_INP_SIZE'] = max_input_size #type_.particle.itemsize

        # The integer type used by the system to address promises
        self.meta['ADR_TYPE']  = np.uint32

        # The type used to store per-link meta-data, currently used only
        # for debugging purposes
        self.meta['META_TYPE'] = np.uint32

        # The type used to store the per-link discriminant, identifying the
        # async call that corresponds to the link's data
        if 'OP_DISC_TYPE' not in self.meta:
            self.meta['OP_DISC_TYPE'] = np.uint32

        # The type used to represent the union of all potential inputs
        self.meta['UNION_TYPE']       = np.dtype((np.void,self.meta['MAX_INP_SIZE']))

        # The type used to represent the array of promises held by a link
        self.meta['PROMISE_ARR_TYPE'] = np.dtype((self.meta['UNION_TYPE'],self.meta['GROUP_SIZE']))

        # The work link type, the fundamental type used to organize work
        # in the system
        self.meta['WORK_LINK_TYPE']   = np.dtype([
            ('promises',self.meta['PROMISE_ARR_TYPE']),
            ('next',self.meta['ADR_TYPE']),
            ('meta_data',self.meta['META_TYPE']),
            ('id',self.meta['OP_DISC_TYPE']),
        ])

        # The integer type used to iterate across IO buffers
        self.meta['IO_ITER_TYPE'] = self.meta['ADR_TYPE']

        # The atomic iterator type used to iterate across IO buffers
        self.meta['IO_ATOMIC_ITER_TYPE'] = np.dtype([
            ('value',self.meta['IO_ITER_TYPE']),
            ('limit',self.meta['IO_ITER_TYPE']),
        ])

        # The atomic iterator type used to iterate across IO buffers
        self.meta['IO_BUFFER_TYPE'] = np.dtype([
            ('toggle',     np.bool_),
            ('padding1',   np.uint8),
            ('padding2',   np.uint16),
            ('padding3',   np.uint32),
            ('data_a',     np.uintp),
            ('data_b',     np.uintp),
            ('capacity',   self.meta['IO_ITER_TYPE']),
            ('input_iter', self.meta['IO_ATOMIC_ITER_TYPE']),
            ('output_iter',self.meta['IO_ATOMIC_ITER_TYPE']),
        ])

        # The array of IO buffers used by the event-based method
        self.meta['IO_BUFFER_ARRAY_TYPE'] = np.dtype(
            (np.uintp,(len(self.async_fns),))
        )


        # The size of the array used to house all work links (measured in links)
        if 'WORK_ARENA_SIZE' not in self.meta:
            self.meta['WORK_ARENA_SIZE'] = 65536

        # The type used as a handle to the arena of work links
        self.meta['WORK_ARENA_TYPE']     = np.dtype([('size',np.intp),('links',np.intp)])

        # The number of independent queues maintained per stack frame and per
        # link pool
        if 'QUEUE_COUNT' not in self.meta:
            self.meta['QUEUE_COUNT']     = 8192 #1024
        # The type of value used to represent links
        self.meta['QUEUE_TYPE']          = np.intp
        # The type used to hold a series of work queues
        self.meta['WORK_POOL_ARR_TYPE']  = np.dtype((
            self.meta['QUEUE_TYPE'],
            (self.meta['QUEUE_COUNT'],)
        ))

        # The struct type that wraps a series of work queues
        self.meta['WORK_POOL_TYPE']      = np.dtype([
            ('queues',self.meta['WORK_POOL_ARR_TYPE'])
        ])
        # The type used to represent
        self.meta['WORK_FRAME_TYPE']     = np.dtype([
            ('children_residents',np.uint32),
            ('padding',np.uintp),
            ('pool',self.meta['WORK_POOL_TYPE'])
        ])

        # The size of the stack (only size-one stacks are currently supported)
        self.meta['STACK_SIZE']         = 1
        # The type used to hold a series of frames
        self.meta['FRAME_ARR_TYPE']     = np.dtype((
            self.meta['WORK_FRAME_TYPE'],
            (self.meta['STACK_SIZE'],)
        ))
        # The type used to hold a work stack and it's associated meta-data
        self.meta['WORK_STACK_TYPE']    = np.dtype([
            ('checkout',    np.uint32),
            ('status_flags',np.uint32),
            ('depth_live',  np.uint32),
            ('frames',      self.meta['FRAME_ARR_TYPE'])
        ])

        # The device context type, tracking the arena, how much of the arena has
        # been claimed the 'easy' way, the pool for allocating/deallocating work links
        # in the arena, and the stack for tracking work links that contain outstanding
        # promises waiting for processing
        self.meta['DEV_CTX_TYPE'] = {}

        self.meta['DEV_CTX_TYPE']["Harmonize"] = np.dtype([
            ('claim_count',np.intp),
            ('arena',      self.meta['WORK_ARENA_TYPE']),
            ('pool' ,      np.intp),
            ('stack',      np.intp)
        ])

        self.meta['DEV_CTX_TYPE']["Event"] = np.dtype([
            ('checkout',   np.uintp),
            ('load_margin',np.uint32),
            ('padding',    np.uint32),
            ('event_io',   self.meta['IO_BUFFER_ARRAY_TYPE']),
        ])

        return self.meta



    # Generates the CUDA/C++ code specifying the program and its rutimme's
    # meta-parameters
    def generate_specification_code(
            self,
            # Whether or not the async functions should be inlined through ptx
            # (currently, this feature is NOT supported)
            inline=False
        ):


        # Accumulator for type definitions
        type_defs = ""

        # A map to store the set of parameter size/alignment specifications
        param_specs = {}

        # Add in parameter specs from the basic async functions
        for func in self.async_fns:
            for kind in fn_arg_ano_list(func):
                if isinstance(kind,numba.types.Record):
                    size      = kind.size
                    align     = alignment(kind)
                    param_specs[(size,align)] = ()

        # Add in parameter specs from the required async functions
        state_kinds = [self.dev_state, self.grp_state, self.thd_state]
        for kind in state_kinds:
                if isinstance(kind,numba.types.Record):
                    size      = kind.size
                    align     = alignment(kind)
                    param_specs[(size,align)] = ()


        # An accumulator for parameter type declarations
        param_decls = ""

        # A map from alignment sizes to their corresponding element type
        element_map = {
            1 : "unsigned char",
            2 : "unsigned short int",
            4 : "unsigned int",
            8 : "unsigned long long int"
        }

        # Create types matching the required size and alignment. Until reliable alignment
        # deduction is implemented, an alignment of 8 will always be used.
        for size, align in param_specs.keys():
            align = 8
            count =  (size + (align-1)) // align
            size  = ((size + (align-1)) // align) * align
            param_decls += "struct _"+str(size)+"b"+str(align) \
                        +" { "+element_map[align]+" data["+str(count)+"]; };\n"



        # Accumulator for prototypes of extern definitions of async functions, initialized
        # with prototypes corresponding to the required async functions
        proto_decls = ""                                                             \
            + "extern \"C\" __device__ int _initialize(void*, void* prog);\n"        \
            + "extern \"C\" __device__ int _finalize  (void*, void* prog);\n"        \
            + "extern \"C\" __device__ int _make_work (bool* result, void* prog);\n"

        # Generate and accumulate prototypes for other async function definitons
        for func in self.async_fns:
            param_text = func_param_text(func,self.type_map,rec_mode="void_ptr",prior=True,clip=0,first_void=True)
            return_type = "void"
            if 'return' in func.__annotations__:
                return_type = map_type_name(self.type_map, func.__annotations__['return'])
            proto_decls += "extern \"C\" __device__ int _"+func.__name__ \
                        +  "("+return_type+"*"+param_text+");\n"

        # Accumulator for async function definitions
        async_defs = ""

        # Forward-declare the structs used to house the definitons
        for func in self.async_fns:
            async_defs += "struct " + pascal_case(func.__name__) + ";\n"

        # Generate an async function struct (with appropriate members)
        # for each async function provided, either attempting to inline the
        # function definition (this currently does not work), or inserting
        # a call to a matching extern function, allowing the scheduling
        # program to jump to the appropriate logic
        for func in self.async_fns:
            async_defs += harm_async_func(func,self.function_map,self.type_map,inline)

        # The metaparameters of the program specification, indicating the sizes
        # of the data structures used
        metaparams = {
            "STASH_SIZE" : self.meta['STASH_SIZE'],
            "FRAME_SIZE" : self.meta['QUEUE_COUNT'],
            "POOL_SIZE"  : self.meta['QUEUE_COUNT'],
        }

        # The declarations of each of the metaparameters
        meta_defs = "".join(["\tstatic const size_t "+name+" = "+str(value) +";\n" for name,value in metaparams.items()])

        # The declaration of the union of all input types
        union_def = "\ttypedef OpUnion<"+",".join([pascal_case(str(func.__name__)) for func in self.async_fns])+"> OpSet;\n"

        # The definition of the device, group, and thread states
        state_defs = "\ttypedef "+map_type_name(self.type_map,self.dev_state,rec_mode="ptr")+" DeviceState;\n" \
                   + "\ttypedef "+map_type_name(self.type_map,self.grp_state,rec_mode="ptr")+" GroupState;\n"  \
                   + "\ttypedef "+map_type_name(self.type_map,self.thd_state,rec_mode="ptr")+" ThreadState;\n"

        # The base function definitions
        spec_def = "struct " + self.spec_name + "{\n"                                                     \
                 + meta_defs + union_def + state_defs                                                     \
                 + harm_template_func(self.init_fn  ,"initialize",self.function_map,self.type_map,inline,True) \
                 + harm_template_func(self.final_fn ,"finalize"  ,self.function_map,self.type_map,inline,True) \
                 + harm_template_func(self.source_fn,"make_work" ,self.function_map,self.type_map,inline,True) \
                 + "};\n"

        return type_defs + param_decls + proto_decls + async_defs + spec_def



    # Returns the CUDA/C++ code specializing the specification for a program type
    def generate_specialization_code(self,kind,shorthand):
        # String template to alias the appropriate specialization to a convenient name
        spec_decl_template = "typedef {kind}Program<{name}> {short_name};\n"

        # String template for the program initialization wrapper
        init_template = ""                                              \
        "extern \"C\"\n"                                                \
        "void init_program(\n"                                          \
        "\tvoid *_dev_ctx_arg,\n"                                       \
        "\tvoid *device_arg,\n"                                         \
        "\tint   grid_size,\n"                                          \
        "\tint   block_size\n"                                          \
        ") {{\n"                                                        \
        "\tauto _dev_ctx = (typename {short_name}::DeviceContext*) _dev_ctx_arg;\n" \
        "\tauto device   = (typename {short_name}::DeviceState) device_arg;\n"      \
        "\t_dev_init<{short_name}><<<grid_size,block_size>>>(*_dev_ctx,device);\n"  \
        "}}\n"

        # String template for the program execution wrapper
        exec_template = ""                                              \
        "extern \"C\"\n"                                                \
        "void exec_program(\n"                                          \
        "\tvoid   *_dev_ctx_arg,\n"                                     \
        "\tvoid   *device_arg,\n"                                       \
        "\tsize_t  cycle_count,\n"                                      \
        "\tint     grid_size,\n"                                        \
        "\tint     block_size\n"                                        \
        ") {{\n"                                                        \
        "\tauto _dev_ctx = (typename {short_name}::DeviceContext*) _dev_ctx_arg;\n"            \
        "\tauto device   = (typename {short_name}::DeviceState) device_arg;\n"                 \
        "\t_dev_exec<{short_name}><<<grid_size,block_size>>>(*_dev_ctx,device,cycle_count);\n" \
        "}}\n"
        #"\tprintf(\"<ctx%p>\",_dev_ctx);\n"\
        #"\tprintf(\"<sta%p>\",device);\n"\

        alloc_state_template = ""                                       \
        "extern \"C\"\n"                                                \
        "void alloc_program(\n"                                         \
        "\tvoid   *_dev_ctx_arg,\n"                                     \
        "\tvoid   *device_arg,\n"                                       \
        "\tsize_t  cycle_count,\n"                                      \
        "\tint     grid_size,\n"                                        \
        "\tint     block_size\n"                                        \
        ") {{\n"                                                        \
        "\tauto _dev_ctx = (typename {short_name}::DeviceContext*) _dev_ctx_arg;\n"            \
        "\tauto device   = (typename {short_name}::DeviceState) device_arg;\n"                 \
        "\t_dev_exec<{short_name}><<<grid_size,block_size>>>(*_dev_ctx,device,cycle_count);\n" \
        "}}\n"


        # String template for async function dispatches
        dispatch_template = ""                                              \
        "extern \"C\" __device__ \n"                                        \
        "int dispatch_{fn}_{kind}(void*{params}){{\n"                       \
        "\t(({short_name}*)fn_param_1)->template {kind}<{fn_type}>({args});\n"  \
        "\treturn 0;\n"                                                     \
        "}}\n"
        #"\tprintf(\"{{ {fn} wrapper }}\");\n"                            \

        # String template for the program execution wrapper
        fn_query_template = ""                                                 \
        "extern \"C\" __device__ "                                          \
        "int query_{field}(void *result, void *prog){{\n"                   \
        "\t(*({kind}*)result) = {prefix}(({short_name}*)prog)->template {field}<{fn_type}>();\n"\
        "\treturn 0;\n"                                                     \
        "}}\n"

        # String template for the program execution wrapper
        query_template = ""                                                 \
        "extern \"C\" __device__ "                                          \
        "int query_{field}(void *result, void *prog){{\n"                   \
        "\t(*({kind}*)result) = {prefix}(({short_name}*)prog)->{field}();\n"\
        "\treturn 0;\n"                                                     \
        "}}\n"


        # String template for field accessors
        accessor_template = ""                                              \
        "extern \"C\" __device__ \n"                                        \
        "int access_{field}(void* result, void* prog){{\n"                  \
        "\t(*(void**)result) = {prefix}(({short_name}*)prog)->{field};\n"   \
        "\treturn 0;\n"                                                     \
        "}}\n"
        #"\tprintf(\"{{ {field} accessor }}\");\n"                          \
        #"\tprintf(\"{{prog %p}}\",prog);\n"                                \
        #"\tprintf(\"{{field%p}}\",*(void**)result);\n"                     \

        # The set of fields that should have accessors, each annotated with
        # the code (if any) that should prefix references to those fields.
        # This is mainly useful for working with references.
        program_fields = [
            (  "device", ""), (   "group", ""), (  "thread", ""),
            #("_dev_ctx","&"), ("_grp_ctx","&"), ("_thd_ctx","&")
        ]


        # Accumulator for includes and initial declarations/typedefs
        preamble      = ""
        preamble     += "#include \""+self.spec_name+".cu\"\n"

        # Accumulator for init/exec/async/sync wrappers
        dispatch_defs = ""
        # Accumulator for accessors
        accessor_defs = ""

        # The name used to refer to the program template specialization
        short_name = self.spec_name.lower() +"_"+shorthand
        # Typedef the specialization to a more convenient shothand
        preamble += spec_decl_template.format(kind=kind,name=self.spec_name,short_name=short_name)

        # Generate the wrappers for kernel entry points
        dispatch_defs += init_template.format(short_name=short_name,name=self.spec_name,kind=kind)
        dispatch_defs += exec_template.format(short_name=short_name,name=self.spec_name,kind=kind)

        # Generate the dispatch functions for each async function
        for fn in self.async_fns:
            # Accepts record parameters as void pointers
            param_text = func_param_text(fn,self.type_map,rec_mode="void_ptr",prior=True,clip=0,first_void=True)
            # Casts the void pointers of parameters to types with matching size and alignment
            arg_text   = func_arg_text(fn,self.type_map,rec_mode="cast_deref",prior=False,clip=1)
            # Generates a wrapper for both async and sync dispatches
            for kind in ["async","sync"]:
                dispatch_defs += dispatch_template.format(
                    short_name=short_name,
                    fn=fn.__name__,
                    fn_type=pascal_case(fn.__name__),
                    params=param_text,
                    args=arg_text,
                    kind=kind,
                )

        # Creates a field accesing function for each field
        for (field,prefix) in program_fields:
            accessor_defs += accessor_template.format(short_name=short_name,field=field,prefix=prefix)

        #fn_query_defs = ""
        #fn_query_list = [ ("load_fraction","float", "") ]
        #for (field,kind,prefix) in fn_query_list:
        #    for fn in self.async_fns:
        #        fn_query_defs += fn_query_template.format(
        #                short_name=short_name,
        #                fn_type=pascal_case(fn.__name__)
        #                field=field,
        #                kind=kind,
        #                prefix=prefix,
        #            )

        #query_defs = ""
        #query_list = [ ]
        #for (field,kind,prefix) in query_list:
        #    query_defs += query_template.format(
        #        short_name=short_name,
        #        field=field,
        #        kind=kind,
        #        prefix=prefix
        #    )

        return preamble + dispatch_defs + accessor_defs #+ fn_query_defs + query_defs



    # Generates the CUDA/C++ code specifying the structure of the program, for later
    # specialization to a specific program type, and compiles the ptx for each async
    # function supplied to the specfication. Both this cuda code and the ptx are
    # saved to the `__ptxcache__` directory for future re-use.
    def generate_code(self):

        # Folder used to cache cuda and ptx code
        cache_path = "__ptxcache__/"

        makedirs(cache_path,exist_ok=True)
        # Generate and save generic program specification
        base_code = self.generate_specification_code()
        base_filename = cache_path+self.spec_name
        base_file = open(base_filename+".cu",mode='w')
        base_file.write(base_code)
        base_file.close()

        # Compile the async function definitions to ptx

        # The list of required async functions
        base_fns = [self.init_fn, self.final_fn, self.source_fn]
        # The full list of async functions
        comp_list = [fn for fn in base_fns] + self.async_fns


        # A list to record the files containing the definitions
        # of each async function definition
        self.fn_def_link_list = []

        # Compile each user-provided function defintion to ptx
        # and save it to an appropriately named file
        for fn in comp_list:
            file_name = fn.__name__+".ptx"
            if fn in base_fns:
                file_name = self.spec_name + "_" + file_name
            file_name = cache_path + file_name
            ptx_file  = open(file_name,mode='w')
            ptx_file.write(extern_device_ptx(fn,self.type_map))
            # Record the path of the generated ptx
            self.fn_def_link_list.append(file_name)
            ptx_file.close()

        self.fn = {}

        # Generate and compile specializations of the specification for
        # each kind of runtime
        for kind, shortname in [("Harmonize","hrm"), ("Event","evt")]:

            self.fn[kind] = {}

            # Generate the cuda code implementing the specialization
            spec_code = self.generate_specialization_code(kind,shortname)
            # Save the code to an appropriately named file
            spec_filename = cache_path+self.spec_name+"_"+shortname
            spec_file = open(spec_filename+".cu",mode='w')
            spec_file.write(spec_code)
            spec_file.close()

            # Compile the specialization to ptx
            compute_level = native_cuda_compute_level()
            # nvcc dep.cu --shared -o libdep.so --compiler-options -fPIC -g
            comp_cmd = "nvcc "+spec_filename+".cu -arch=compute_"+compute_level+  \
                    " -include "+HARMONIZE_ROOT_CPP+" --shared -o "+spec_filename+".so"+ \
                    " --compiler-options -fPIC"

            if DEBUG:
                comp_cmd += " -g"
            subprocess.run(comp_cmd.split(),shell=False,check=True)
            path = abspath("/home/brax/cuda_lab/libdep.so")
            binding.load_library_permanently(path)

            # Create handles to reference the cuda entry wrapper functions
            void    = numba.types.void
            vp      = numba.types.voidptr
            i32     = numba.types.int32
            usize   = numba.types.uintp
            sig     = numba.core.typing.signature
            ext_fn  = numba.types.ExternalFunction
            context = numba.from_dtype(self.meta['DEV_CTX_TYPE'][kind])
            #print("\n\n\n\n",self.dev_state,"\n\n\n")
            state   = self.dev_state #numba.from_dtype(self.dev_state)

            init_program  = ext_fn("init_program",  sig(void, vp, vp, i32, i32))
            exec_program  = ext_fn("exec_program",  sig(void, vp, vp, usize, i32, i32))
            store_state   = ext_fn("store_state",   sig(void, vp, state))
            load_state    = ext_fn("load_state",    sig(void, state, vp))
            if kind == "Event":
                # IO_SIZE, LOAD_MARGIN
                alloc_context = ext_fn("alloc_context", sig(vp,usize,usize))
            else:
                # ARENA SIZE, POOL SIZE, STACK SIZE
                alloc_context = ext_fn("alloc_context", sig(vp,usize,usize,usize))
            alloc_state   = ext_fn("alloc_state",   sig(vp))
            free_context  = ext_fn("free_context",  sig(void,vp))
            free_state    = ext_fn("free_state",    sig(void,vp))
            rt_alloc      = ext_fn("rt_alloc",      sig(vp, usize))
            rt_free       = ext_fn("rt_free",       sig(void, vp))


            # Finally, compile the entry functions, saving it for later use
            self.fn[kind]['init_program']  = init_program
            self.fn[kind]['exec_program']  = exec_program
            self.fn[kind]['store_state']   = store_state
            self.fn[kind]['load_state']    = load_state
            self.fn[kind]['rt_alloc']      = rt_alloc
            self.fn[kind]['alloc_state']   = alloc_state
            self.fn[kind]['alloc_context'] = alloc_context
            self.fn[kind]['free_state']    = free_state
            self.fn[kind]['free_context']  = free_context
            self.fn[kind]['rt_free']       = rt_free


    # Returns a HarmonizeRuntime instance based off of the program specification
    def harmonize_fns(self):
        return self.fn["Harmonize"]


    # Returns an EventRuntime instance based off of the program specification, using
    # buffers of size `io_capacity` to store intermediate data for each event type and
    # halting work generation when the space left in any buffer is less than or equal to
    # `load_margin`.
    def event_instance(self):
        return self.fn["Event"]




    # Injects the async/sync dispatch functions and state accessors into the
    # global scope of the caller. This should only be done if you are sure that
    # the injection of these fields into the global namespace of the calling
    # module won't overwrite anything. While convenient in some respects, this
    # function also gives no indication to linters that the corresponding fields
    # will be injected, leading to linters incorrectly (though understandably)
    # marking the fields as undefined
    def inject_fns(state_spec,async_fns):

        dev_state, grp_state, thd_state = state_spec

        for func in async_fns:
            sig = fn_sig(func)
            name = func.__name__
            for kind in ["async","sync"]:
                dispatch_fn = cuda.declare_device("dispatch_"+name+"_"+kind, sig)
                inject_global(kind+"_"+name,dispatch_fn,1)

        field_list = [
            ("device",dev_state),
            ("group",grp_state),
            ("thread",thd_state),
        ]
        for name, kind in field_list:
            sig = kind(numba.uintp)
            access_fn = cuda.declare_device("access_"+name,sig)
            inject_global(name,access_fn,1)

    # Returns the `device`, `group`, and `thread` accessor function
    # handles of the specification as a triplet
    def access_fns(state_spec):

        dev_state, grp_state, thd_state = state_spec

        field_list = [
            ("device",dev_state),
            ("group",grp_state),
            ("thread",thd_state),
        ]

        result = []
        for name, kind in field_list:
            sig = kind(numba.uintp)
            result.append(cuda.declare_device("access_"+name,sig))
        return tuple(result)

    # Returns the async/sync function handles for the supplied functions, using
    # `kind` to switch between async and sync
    def dispatch_fns(kind,*async_fns):
        async_fns, = async_fns
        result = []
        for func in async_fns:
            sig = fn_sig(func)
            name = func.__name__
            result.append(cuda.declare_device("dispatch_"+name+"_"+kind, sig))
        return tuple(result)

    #def query(kind,*fields):
    #    fields, = fields
    #    result = []
    #    for field in fields:
    #        result.append(cuda.declare_device("dispatch_"+name+"_"+kind, sig))
    #    return tuple(result)


    # Returns the handles for the on-gpu functions used to asynchronously schedule
    # the supplied function
    def async_dispatch(*async_fns):
        return RuntimeSpec.dispatch_fns("async",async_fns)

    # Returns the handles for the on-gpu functions used to immediately call
    # the supplied function
    def sync_dispatch(*async_fns):
        return RuntimeSpec.dispatch_fns("sync",async_fns)


