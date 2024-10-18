(function(t,n){typeof exports=="object"&&typeof module<"u"?n(exports):typeof define=="function"&&define.amd?define(["exports"],n):(t=typeof globalThis<"u"?globalThis:t||self,n(t.breasties_2={}))})(this,function(t){"use strict";const i={yourCustomWidget:{emits:["click","change"],props:{attribute_name:{type:String},js_attr_name:{type:String}},setup(o,{emit:e}){return{btnClick:"margin: 0 10px;background: #F48FB1 !important;",btnChange:"margin: 0 10px;#CE93D8 !important;",triggerClick:()=>e("click"),triggerChange:()=>e("change")}},template:`
        <div>
            <v-btn @click="triggerClick" :style="btnClick">Custom click</v-btn>
            <v-btn @click="triggerChange" :style="btnChange">Custom change</v-btn>
        </div>`}};function c(o){Object.keys(i).forEach(e=>{o.component(e,i[e])})}t.install=c,Object.defineProperty(t,Symbol.toStringTag,{value:"Module"})});
